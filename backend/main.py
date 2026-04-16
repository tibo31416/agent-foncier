"""
Agent Foncier — Backend FastAPI v2.1
Déploiement : Railway
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import urllib.parse
import os
import json
import re
import io
import anthropic
from pypdf import PdfReader

app = FastAPI(title="Agent Foncier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PDF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.geoportail-urbanisme.gouv.fr/",
}


# ── Modèles ───────────────────────────────────────────────────────

class ParcelleRequest(BaseModel):
    code_insee: str
    section: str
    numero: str
    com_abs: str = "000"

class BatimentRequest(BaseModel):
    idu: str

class AnalyseRequest(BaseModel):
    pdf_url: str
    idu: str
    surface_parcelle: int
    zone_plu: str
    commune: str
    batiment: dict = {}


# ── Helper : télécharger + extraire texte PDF ─────────────────────

def extraire_page(page) -> str:
    """Extrait le texte d'une page PDF avec layout si possible."""
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return page.extract_text() or ""


async def telecharger_et_extraire_pdf(pdf_url: str, zone: str, max_chars: int = 15000) -> dict:
    """
    Télécharge le PDF règlement PLU et extrait le texte pertinent pour la zone.
    HTTP/2 obligatoire pour data.geopf.fr.
    """
    async with httpx.AsyncClient(
        timeout=60,
        follow_redirects=True,
        http2=True,
        headers=PDF_HEADERS,
    ) as client:
        r = await client.get(pdf_url)

    if r.status_code != 200:
        return {
            "succes": False,
            "erreur": f"HTTP {r.status_code} lors du téléchargement",
            "texte": "", "nb_pages": 0, "pages_zone": [],
        }

    if len(r.content) < 1000:
        return {
            "succes": False,
            "erreur": "Fichier PDF vide ou trop petit",
            "texte": "", "nb_pages": 0, "pages_zone": [],
        }

    try:
        reader = PdfReader(io.BytesIO(r.content))
        nb_pages = len(reader.pages)
        zone_upper = zone.upper()

        zone_pages = []
        general_pages = []

        for i, page in enumerate(reader.pages):
            t = extraire_page(page)
            t_upper = t.upper()

            # Pages spécifiques à la zone (articles, sections)
            if (
                f"ZONE {zone_upper}" in t_upper
                or f"ARTICLE {zone_upper}" in t_upper
                or f"SECTION {zone_upper}" in t_upper
                or f"{zone_upper}.1" in t_upper
                or f"{zone_upper}.2" in t_upper
                or f"{zone_upper}.3" in t_upper
                or f"\n{zone_upper} -" in t_upper
                or f"\n{zone_upper}\n" in t_upper
            ):
                zone_pages.append((i, t))

            # Dispositions générales (premières pages)
            elif i < 20 and any(
                kw in t_upper
                for kw in ["DISPOSITION GÉNÉRALE", "DISPOSITION GENERALE", "SOMMAIRE", "DÉFINITION"]
            ):
                general_pages.append((i, t))

        # Ajouter la page suivante de chaque page zone (continuité des articles)
        pages_zone_idx = {p[0] for p in zone_pages}
        pages_contexte = []
        for idx, _ in zone_pages:
            if idx + 1 < nb_pages and (idx + 1) not in pages_zone_idx:
                pages_contexte.append((idx + 1, extraire_page(reader.pages[idx + 1])))

        # Assembler : générales + zone + contexte immédiat
        combined = ""
        for _, t in general_pages[:2]:
            combined += t + "\n"
        for _, t in zone_pages + pages_contexte:
            combined += t + "\n"
            if len(combined) > max_chars:
                break

        # Fallback : aucune page trouvée → premières pages du doc
        if not zone_pages:
            combined = ""
            for i in range(min(20, nb_pages)):
                combined += extraire_page(reader.pages[i]) + "\n"
                if len(combined) > max_chars:
                    break

        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n[... tronqué à 15 000 caractères ...]"

        return {
            "succes": True,
            "texte": combined,
            "nb_pages": nb_pages,
            "pages_zone": [p[0] for p in zone_pages],
            "taille_ko": len(r.content) // 1024,
        }

    except Exception as e:
        return {
            "succes": False,
            "erreur": f"Erreur lecture PDF : {e}",
            "texte": "", "nb_pages": 0, "pages_zone": [],
        }


# ── Route 1 : Parcelle ────────────────────────────────────────────

@app.post("/parcelle")
async def get_parcelle(req: ParcelleRequest):
    section = req.section.upper()
    numero  = req.numero.zfill(4)
    com_abs = req.com_abs.zfill(3)

    params = {
        "code_insee": req.code_insee,
        "section":    section,
        "numero":     numero,
        "com_abs":    com_abs,
    }
    cad_url = "https://apicarto.ign.fr/api/cadastre/parcelle?" + urllib.parse.urlencode(params)

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(cad_url)
        r.raise_for_status()
        cad_data = r.json()

    features = cad_data.get("features", [])
    if not features:
        raise HTTPException(status_code=404, detail="Parcelle introuvable.")

    # Communes absorbées → retourner les choix
    if len(features) > 1:
        return {
            "status": "ambigue",
            "message": f"{len(features)} parcelles trouvées. Précisez com_abs.",
            "choix": [
                {
                    "idu":     f["properties"]["idu"],
                    "surface": f["properties"]["contenance"],
                    "commune": f["properties"]["nom_com"],
                    "com_abs": f["properties"]["com_abs"],
                    "section": f["properties"]["section"],
                    "numero":  f["properties"]["numero"],
                }
                for f in features
            ],
        }

    props = features[0]["properties"]
    geom  = features[0]["geometry"]
    idu   = props["idu"]

    # GPU : zone PLU + URL PDF
    geom_encoded = urllib.parse.quote(json.dumps(geom))
    zone_plu = type_zone = libelle_zone = pdf_url = nom_document = type_document = idurba = None

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r_zone = await client.get(
                f"https://apicarto.ign.fr/api/gpu/zone-urba?geom={geom_encoded}"
            )
            if r_zone.status_code == 200:
                zones = r_zone.json().get("features", [])
                if zones:
                    z = zones[0]["properties"]
                    zone_plu     = z.get("libelle")
                    type_zone    = z.get("typezone")
                    libelle_zone = z.get("libelong", "")
                    idurba       = z.get("idurba")
                    nomfic       = z.get("nomfic", "")
                    urlfic       = z.get("urlfic", "")
                    partition    = z.get("partition", "")

                    r_doc = await client.get(
                        f"https://apicarto.ign.fr/api/gpu/document?geom={geom_encoded}"
                    )
                    if r_doc.status_code == 200:
                        docs = r_doc.json().get("features", [])
                        if docs:
                            d            = docs[0]["properties"]
                            nom_document = d.get("grid_title")
                            type_document = d.get("du_type")
                            gpu_doc_id   = d.get("gpu_doc_id", "")
                            pdf_url = urlfic if urlfic else (
                                f"https://data.geopf.fr/annexes/gpu/documents/{partition}/{gpu_doc_id}/{nomfic}"
                                if gpu_doc_id and partition and nomfic else None
                            )
        except Exception:
            pass

    return {
        "status":          "ok",
        "idu":             idu,
        "surface_parcelle": props["contenance"],
        "commune":         props["nom_com"],
        "code_commune":    props["code_insee"],
        "section":         props["section"],
        "numero":          props["numero"],
        "com_abs":         props["com_abs"],
        "code_arr":        props.get("code_arr"),
        "geometrie":       geom,
        "plu": {
            "zone":          zone_plu,
            "type_zone":     type_zone,
            "libelle_zone":  libelle_zone,
            "idurba":        idurba,
            "nom_document":  nom_document,
            "type_document": type_document,
            "pdf_url":       pdf_url,
        },
    }


# ── Route 2 : Bâtiment (BDNB) ─────────────────────────────────────

@app.post("/batiment")
async def get_batiment(req: BatimentRequest):
    idu  = req.idu
    BDNB = "https://api.bdnb.io/v1/bdnb/donnees"

    async with httpx.AsyncClient(timeout=20) as client:
        # rel_batiment_groupe_parcelle est indexé sur parcelle_id → rapide
        r_rel = await client.get(
            f"{BDNB}/rel_batiment_groupe_parcelle"
            f"?parcelle_id=eq.{idu}&select=batiment_groupe_id&limit=5",
            headers={"Accept": "application/json"},
        )

        if r_rel.status_code != 200 or not r_rel.json():
            return {"status": "not_found", "message": "Aucun bâtiment BDNB trouvé.", "idu": idu}

        batiment_groupe_id = r_rel.json()[0]["batiment_groupe_id"]

        r_bat = await client.get(
            f"{BDNB}/batiment_groupe_complet"
            f"?batiment_groupe_id=eq.{batiment_groupe_id}"
            f"&select=batiment_groupe_id,s_geom_groupe,nb_niveau,annee_construction,"
            f"usage_niveau_1_txt,mat_mur_txt,mat_toit_txt,nb_log,"
            f"libelle_adr_principale_ban,classe_bilan_dpe&limit=1",
            headers={"Accept": "application/json"},
        )

        if r_bat.status_code != 200 or not r_bat.json():
            return {"status": "partial", "batiment_groupe_id": batiment_groupe_id}

        bat = r_bat.json()[0]
        return {
            "status":             "ok",
            "idu":                idu,
            "batiment_groupe_id": batiment_groupe_id,
            "surface_au_sol":     bat.get("s_geom_groupe"),
            "nb_niveaux":         bat.get("nb_niveau"),
            "annee_construction": bat.get("annee_construction"),
            "usage":              bat.get("usage_niveau_1_txt"),
            "materiaux_mur":      bat.get("mat_mur_txt"),
            "materiaux_toit":     bat.get("mat_toit_txt"),
            "nb_logements":       bat.get("nb_log"),
            "adresse":            bat.get("libelle_adr_principale_ban"),
            "classe_dpe":         bat.get("classe_bilan_dpe"),
        }


# ── Route 3 : Analyse Claude ──────────────────────────────────────

@app.post("/analyse")
async def analyse_plu(req: AnalyseRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Clé Anthropic manquante.")

    # Étape A : télécharger et extraire le texte du PDF
    pdf_result = await telecharger_et_extraire_pdf(req.pdf_url, req.zone_plu)

    if pdf_result["succes"]:
        source_info = (
            f"PDF téléchargé et lu directement "
            f"({pdf_result['taille_ko']} Ko, {pdf_result['nb_pages']} pages, "
            f"pages zone extraites : {pdf_result['pages_zone']})"
        )
        instruction_source = f"""
Voici le texte extrait directement du règlement PLU officiel :

=== DÉBUT RÈGLEMENT PLU ===
{pdf_result['texte']}
=== FIN RÈGLEMENT PLU ===

Base ton analyse UNIQUEMENT sur ce texte. Cite les numéros d'articles (ex: UG.3.1) quand ils sont présents.
"""
    else:
        source_info = f"PDF non accessible : {pdf_result.get('erreur')}"
        instruction_source = f"""
Le PDF n'a pas pu être téléchargé ({pdf_result.get('erreur')}).
Analyse basée sur tes connaissances générales de la zone {req.zone_plu} — précise-le clairement.
"""

    # Étape B : contexte bâtiment existant
    bat = req.batiment
    if bat and bat.get("status") == "ok":
        contexte_batiment = (
            f"BÂTIMENT EXISTANT (BDNB) :\n"
            f"- Surface au sol : {bat.get('surface_au_sol')} m²\n"
            f"- Niveaux : {bat.get('nb_niveaux')}\n"
            f"- Année : {bat.get('annee_construction')}\n"
            f"- Usage : {bat.get('usage')}\n"
            f"- Matériaux murs : {bat.get('materiaux_mur')}\n"
            f"- Logements : {bat.get('nb_logements')}"
        )
    else:
        contexte_batiment = "Aucun bâtiment existant détecté (parcelle non bâtie ou données indisponibles)."

    # Étape C : appel Claude
    prompt = f"""Tu es un expert en droit de l'urbanisme français et en promotion immobilière.

PARCELLE :
- IDU : {req.idu}
- Commune : {req.commune}
- Surface : {req.surface_parcelle} m²
- Zone PLU : {req.zone_plu}
- Source : {source_info}

{contexte_batiment}

{instruction_source}

Réponds UNIQUEMENT en JSON valide, sans texte avant ni après :

{{
  "zone": "{req.zone_plu}",
  "type_zone": "U/AU/A/N",
  "destination_principale": "description courte",
  "ce_qui_est_autorise": ["usage 1", "usage 2"],
  "ce_qui_est_interdit": ["interdit 1", "interdit 2"],
  "regles_hauteur": {{
    "hauteur_max": "valeur avec unité ou null",
    "regles_specifiques": "description précise avec numéro d'article si disponible"
  }},
  "regles_implantation": {{
    "retrait_rue": "valeur ou règle",
    "retrait_limites": "valeur ou règle",
    "emprise_sol_max": "% ou null"
  }},
  "surface_constructible": {{
    "shon_estimee": "estimation en m² pour cette parcelle de {req.surface_parcelle} m²",
    "nb_niveaux_max": "nombre ou null",
    "methode_calcul": "explication"
  }},
  "stationnement": "règles applicables",
  "performances_energetiques": "RE2020 ou autre exigence",
  "points_attention": ["point 1", "point 2", "point 3"],
  "opportunite_promoteur": {{
    "score": 65,
    "commentaire": "évaluation synthétique"
  }},
  "source_analysee": "{req.pdf_url}"
}}
"""

    client_ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        message = client_ai.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        full_text = "".join(b.text for b in message.content if b.type == "text")
        full_text = re.sub(r"```json\s*", "", full_text)
        full_text = re.sub(r"```\s*", "", full_text).strip()

        json_match = re.search(r"\{[\s\S]+\}", full_text)
        analyse = json.loads(json_match.group()) if json_match else {
            "erreur": "JSON invalide", "brut": full_text[:500]
        }

        return {
            "status":         "ok",
            "pdf_telecharge": pdf_result["succes"],
            "pdf_info": {
                "taille_ko":  pdf_result.get("taille_ko"),
                "nb_pages":   pdf_result.get("nb_pages"),
                "pages_zone": pdf_result.get("pages_zone"),
            },
            "analyse": analyse,
        }

    except json.JSONDecodeError as e:
        return {"status": "parse_error", "message": str(e), "brut": full_text[:500]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Claude : {e}")


# ── Health check ──────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agent Foncier API", "version": "2.1"}
