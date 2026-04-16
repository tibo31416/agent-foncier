"""
Agent Foncier — Backend FastAPI
Déploiement : Railway (gratuit)
Routes :
  POST /parcelle  → IGN Cadastre + GPU → IDU, surface, zone PLU, URL PDF règlement
  POST /batiment  → BDNB → surface au sol, nb niveaux, année construction, usage
  POST /analyse   → Claude → analyse du règlement PLU
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import urllib.parse
import os
import json
import re
import anthropic

app = FastAPI(title="Agent Foncier API")

# CORS — autorise ton frontend Vercel (et localhost pour le dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod, remplace * par ton URL Vercel exacte
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clé Anthropic — à mettre dans les variables d'environnement Railway
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ── Modèles de requête ────────────────────────────────────────────

class ParcelleRequest(BaseModel):
    code_insee: str          # ex: "75056"
    section: str             # ex: "DW"
    numero: str              # ex: "0241"
    com_abs: str = "000"     # ex: "000" (par défaut, "823" pour Toulouse)

class BatimentRequest(BaseModel):
    idu: str                 # ex: "75113000DW0241"

class AnalyseRequest(BaseModel):
    pdf_url: str             # URL du PDF règlement PLU
    idu: str
    surface_parcelle: int    # m²
    zone_plu: str            # ex: "UG"
    commune: str
    batiment: dict = {}      # données BDNB (peut être vide)


# ── Helpers ───────────────────────────────────────────────────────

async def geocode_point_to_parcel(lon: float, lat: float) -> dict | None:
    """Intersection GPS → parcelle via API Carto IGN"""
    geom = json.dumps({"type": "Point", "coordinates": [lon, lat]})
    url = f"https://apicarto.ign.fr/api/cadastre/parcelle?geom={urllib.parse.quote(geom)}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        return features[0] if features else None


async def buffer_search(lon: float, lat: float, delta: float) -> dict | None:
    """Buffer polygon autour d'un point → première parcelle trouvée"""
    geom = json.dumps({
        "type": "Polygon",
        "coordinates": [[
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta],
        ]]
    })
    url = f"https://apicarto.ign.fr/api/cadastre/parcelle?geom={urllib.parse.quote(geom)}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        if features:
            # Trier par surface croissante (la plus petite = la plus proche)
            features.sort(key=lambda f: f["properties"].get("contenance", 9999))
            return features[0]
        return None


# ── Route 1 : Parcelle ────────────────────────────────────────────

@app.post("/parcelle")
async def get_parcelle(req: ParcelleRequest):
    """
    Identifie une parcelle depuis sa référence cadastrale.
    Retourne : IDU, surface, géométrie, zone PLU, URL PDF règlement.
    """
    section = req.section.upper()
    numero = req.numero.zfill(4)
    com_abs = req.com_abs.zfill(3)

    # 1. Appel API Carto Cadastre
    params = {
        "code_insee": req.code_insee,
        "section": section,
        "numero": numero,
        "com_abs": com_abs,
    }
    cad_url = "https://apicarto.ign.fr/api/cadastre/parcelle?" + urllib.parse.urlencode(params)

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(cad_url)
        r.raise_for_status()
        cad_data = r.json()

    features = cad_data.get("features", [])
    if not features:
        raise HTTPException(status_code=404, detail="Parcelle introuvable. Vérifiez code INSEE, section et numéro.")

    # Si plusieurs résultats (communes absorbées), on retourne la liste pour le frontend
    if len(features) > 1:
        choix = []
        for f in features:
            p = f["properties"]
            choix.append({
                "idu": p["idu"],
                "surface": p["contenance"],
                "commune": p["nom_com"],
                "com_abs": p["com_abs"],
                "section": p["section"],
                "numero": p["numero"],
            })
        return {
            "status": "ambigue",
            "message": f"{len(features)} parcelles trouvées (communes absorbées). Précisez com_abs.",
            "choix": choix
        }

    feature = features[0]
    props = feature["properties"]
    geom = feature["geometry"]
    idu = props["idu"]

    # 2. GPU : zone urbanisme + document PLU
    geom_encoded = urllib.parse.quote(json.dumps(geom))
    zone_plu = None
    type_zone = None
    libelle_zone = None
    pdf_url = None
    nom_document = None
    type_document = None
    idurba = None

    async with httpx.AsyncClient(timeout=15) as client:
        # Zone urbanisme
        try:
            r_zone = await client.get(
                f"https://apicarto.ign.fr/api/gpu/zone-urba?geom={geom_encoded}"
            )
            if r_zone.status_code == 200:
                zones = r_zone.json().get("features", [])
                if zones:
                    z = zones[0]["properties"]
                    zone_plu = z.get("libelle")
                    type_zone = z.get("typezone")
                    libelle_zone = z.get("libelong", "")
                    idurba = z.get("idurba")
                    nomfic = z.get("nomfic", "")
                    urlfic = z.get("urlfic", "")
                    partition = z.get("partition", "")

                    # Document PLU
                    r_doc = await client.get(
                        f"https://apicarto.ign.fr/api/gpu/document?geom={geom_encoded}"
                    )
                    if r_doc.status_code == 200:
                        docs = r_doc.json().get("features", [])
                        if docs:
                            d = docs[0]["properties"]
                            nom_document = d.get("grid_title")
                            type_document = d.get("du_type")
                            gpu_doc_id = d.get("gpu_doc_id", "")

                            # Construire URL PDF règlement
                            if urlfic:
                                pdf_url = urlfic
                            elif gpu_doc_id and partition and nomfic:
                                pdf_url = f"https://data.geopf.fr/annexes/gpu/documents/{partition}/{gpu_doc_id}/{nomfic}"
        except Exception:
            pass  # GPU indisponible → on continue sans

    return {
        "status": "ok",
        "idu": idu,
        "surface_parcelle": props["contenance"],
        "commune": props["nom_com"],
        "code_commune": props["code_insee"],
        "section": props["section"],
        "numero": props["numero"],
        "com_abs": props["com_abs"],
        "code_arr": props.get("code_arr"),
        "geometrie": geom,
        "plu": {
            "zone": zone_plu,
            "type_zone": type_zone,
            "libelle_zone": libelle_zone,
            "idurba": idurba,
            "nom_document": nom_document,
            "type_document": type_document,
            "pdf_url": pdf_url,
        }
    }


# ── Route 2 : Bâtiment (BDNB) ─────────────────────────────────────

@app.post("/batiment")
async def get_batiment(req: BatimentRequest):
    """
    Cherche les caractéristiques du bâtiment existant sur la parcelle via BDNB.
    Stratégie : rel_batiment_groupe_parcelle (indexé) → batiment_groupe_complet
    Retourne : surface au sol, nb niveaux, année construction, usage, matériaux.
    """
    idu = req.idu
    # Extraire le code département depuis l'IDU (2 ou 3 premiers chars)
    dep = idu[:2]
    if dep in ("97",):  # DOM-TOM → 3 chars
        dep = idu[:3]

    BDNB = "https://api.bdnb.io/v1/bdnb/donnees"

    async with httpx.AsyncClient(timeout=20) as client:
        # Étape 1 : rel_batiment_groupe_parcelle → batiment_groupe_id
        url_rel = (
            f"{BDNB}/rel_batiment_groupe_parcelle"
            f"?parcelle_id=eq.{idu}"
            f"&select=batiment_groupe_id,parcelle_id"
            f"&limit=5"
        )
        r_rel = await client.get(url_rel, headers={"Accept": "application/json"})

        if r_rel.status_code != 200 or not r_rel.json():
            return {
                "status": "not_found",
                "message": "Aucun bâtiment BDNB trouvé pour cette parcelle.",
                "idu": idu,
            }

        rel_data = r_rel.json()
        batiment_groupe_id = rel_data[0]["batiment_groupe_id"]

        # Étape 2 : batiment_groupe_complet → caractéristiques
        url_bat = (
            f"{BDNB}/batiment_groupe_complet"
            f"?batiment_groupe_id=eq.{batiment_groupe_id}"
            f"&select=batiment_groupe_id,s_geom_groupe,nb_niveau,annee_construction,"
            f"usage_niveau_1_txt,mat_mur_txt,mat_toit_txt,nb_log,"
            f"libelle_adr_principale_ban,classe_bilan_dpe"
            f"&limit=1"
        )
        r_bat = await client.get(url_bat, headers={"Accept": "application/json"})

        if r_bat.status_code != 200 or not r_bat.json():
            return {
                "status": "partial",
                "message": "Référence bâtiment trouvée mais données incomplètes.",
                "batiment_groupe_id": batiment_groupe_id,
            }

        bat = r_bat.json()[0]

        return {
            "status": "ok",
            "idu": idu,
            "batiment_groupe_id": batiment_groupe_id,
            "surface_au_sol": bat.get("s_geom_groupe"),       # m²
            "nb_niveaux": bat.get("nb_niveau"),               # nombre de niveaux
            "annee_construction": bat.get("annee_construction"),
            "usage": bat.get("usage_niveau_1_txt"),
            "materiaux_mur": bat.get("mat_mur_txt"),
            "materiaux_toit": bat.get("mat_toit_txt"),
            "nb_logements": bat.get("nb_log"),
            "adresse": bat.get("libelle_adr_principale_ban"),
            "classe_dpe": bat.get("classe_bilan_dpe"),
        }


# ── Route 3 : Analyse Claude ──────────────────────────────────────

@app.post("/analyse")
async def analyse_plu(req: AnalyseRequest):
    """
    Envoie le PDF règlement PLU + contexte à Claude.
    Retourne une analyse structurée : ce qu'on peut construire + infos clés.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Clé Anthropic manquante (variable ANTHROPIC_API_KEY).")

    # Construire le contexte bâtiment existant
    bat = req.batiment
    contexte_batiment = ""
    if bat and bat.get("status") == "ok":
        contexte_batiment = f"""
BÂTIMENT EXISTANT SUR LA PARCELLE (source BDNB) :
- Surface au sol : {bat.get('surface_au_sol', 'N/A')} m²
- Nombre de niveaux : {bat.get('nb_niveaux', 'N/A')}
- Année de construction : {bat.get('annee_construction', 'N/A')}
- Usage actuel : {bat.get('usage', 'N/A')}
- Matériaux murs : {bat.get('materiaux_mur', 'N/A')}
- Logements : {bat.get('nb_logements', 'N/A')}
"""
    else:
        contexte_batiment = "BÂTIMENT EXISTANT : Aucune donnée disponible dans la BDNB."

    prompt = f"""Tu es un expert en droit de l'urbanisme français et en promotion immobilière.

PARCELLE ANALYSÉE :
- IDU : {req.idu}
- Commune : {req.commune}
- Surface parcelle : {req.surface_parcelle} m²
- Zone PLU : {req.zone_plu}
- URL règlement : {req.pdf_url}

{contexte_batiment}

Ta mission : lire le règlement PLU disponible à cette URL et produire une analyse complète.

Réponds UNIQUEMENT en JSON valide, sans texte avant ni après, avec cette structure exacte :

{{
  "zone": "{req.zone_plu}",
  "type_zone": "U/AU/A/N",
  "destination_principale": "description courte",
  "ce_qui_est_autorise": ["liste", "des", "usages", "autorisés"],
  "ce_qui_est_interdit": ["liste", "des", "interdictions", "principales"],
  "regles_hauteur": {{
    "hauteur_max": "valeur ou null",
    "regles_specifiques": "description des règles de hauteur"
  }},
  "regles_implantation": {{
    "retrait_rue": "valeur ou règle",
    "retrait_limites": "valeur ou règle",
    "emprise_sol_max": "valeur en % ou null"
  }},
  "surface_constructible": {{
    "shon_estimee": "estimation en m²",
    "nb_niveaux_max": "nombre ou null",
    "methode_calcul": "explication courte"
  }},
  "stationnement": "règles de stationnement",
  "performances_energetiques": "exigences RE2020 ou autre",
  "points_attention": ["point 1", "point 2", "point 3"],
  "opportunite_promoteur": {{
    "score": 0,
    "commentaire": "évaluation courte"
  }},
  "source_analysee": "{req.pdf_url}"
}}

Si tu ne peux pas accéder au PDF (lien mort, ZIP non lisible), indique-le dans "points_attention" et donne quand même une analyse de base selon le type de zone {req.zone_plu} en contexte français.
"""

    client_ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        message = client_ai.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
            }],
            messages=[{"role": "user", "content": prompt}]
        )

        # Extraire le texte de la réponse (peut être mix text + tool_use)
        full_text = ""
        for block in message.content:
            if block.type == "text":
                full_text += block.text

        # Parser le JSON retourné par Claude
        # Nettoyer les éventuels backticks markdown
        full_text = re.sub(r"```json\s*", "", full_text)
        full_text = re.sub(r"```\s*", "", full_text)
        full_text = full_text.strip()

        # Extraire le JSON (chercher le premier { ... } complet)
        json_match = re.search(r'\{[\s\S]+\}', full_text)
        if json_match:
            analyse = json.loads(json_match.group())
        else:
            analyse = {"erreur": "Claude n'a pas retourné de JSON valide.", "brut": full_text[:500]}

        return {
            "status": "ok",
            "analyse": analyse,
        }

    except json.JSONDecodeError as e:
        return {
            "status": "parse_error",
            "message": f"Erreur parsing JSON Claude: {e}",
            "brut": full_text[:500] if 'full_text' in locals() else "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Claude API: {str(e)}")


# ── Health check ──────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agent Foncier API", "version": "1.0"}
