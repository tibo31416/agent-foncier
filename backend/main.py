"""
Agent Foncier — Backend FastAPI v3.0
Routes :
  POST /parcelle   → IGN Cadastre + GPU → IDU, surface, zone PLU, URL PDF
  POST /batiment   → BDNB → surface au sol, niveaux, année, DPE
  POST /technique  → 16 APIs en parallèle → sol, risques, servitudes, environnement, réseaux
  POST /analyse    → télécharge PDF + intègre données techniques → Claude → JSON structuré
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
import urllib.parse
import os, json, re, io
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
    "Accept":     "application/pdf,*/*",
    "Referer":    "https://www.geoportail-urbanisme.gouv.fr/",
}

GR  = "https://georisques.gouv.fr/api/v1"   # Géorisques
AC  = "https://apicarto.ign.fr/api"          # APICarto IGN
BAN = "https://api-adresse.data.gouv.fr"
GEO = "https://geo.api.gouv.fr"
BDNB = "https://api.bdnb.io/v1/bdnb/donnees"


# ══════════════════════════════════════════════════════════════════
# MODÈLES
# ══════════════════════════════════════════════════════════════════

class ParcelleRequest(BaseModel):
    code_insee: str
    section:    str
    numero:     str
    com_abs:    str = "000"

class BatimentRequest(BaseModel):
    idu: str

class TechniqueRequest(BaseModel):
    idu:        str
    lon:        float
    lat:        float
    code_insee: str
    geom:       dict   # GeoJSON geometry de la parcelle

class AnalyseRequest(BaseModel):
    pdf_url:         str
    idu:             str
    surface_parcelle: int
    zone_plu:        str
    commune:         str
    batiment:        dict = {}
    technique:       dict = {}   # données techniques (optionnel mais enrichit l'analyse)


# ══════════════════════════════════════════════════════════════════
# HELPERS GÉNÉRIQUES
# ══════════════════════════════════════════════════════════════════

async def _safe_get(url: str, retries: int = 2) -> dict | None:
    """
    GET JSON avec retry silencieux.
    Retourne None si échec (503, timeout, etc.) — ne bloque jamais le flux.
    """
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=12,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            ) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 503 and attempt < retries - 1:
                    await asyncio.sleep(1.5)
                    continue
                return None
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(1)
    return None


def _features(r) -> list:
    """Extrait les features d'une réponse GeoJSON."""
    if not r or not isinstance(r, dict):
        return []
    return r.get("features", [])


def _data(r) -> list:
    """Extrait le champ 'data' d'une réponse Géorisques."""
    if not r or not isinstance(r, dict):
        return []
    return r.get("data", [])


def _first(r, key: str, source="data"):
    """Récupère un champ dans le premier élément de data[] ou features[]."""
    items = _data(r) if source == "data" else _features(r)
    if items and isinstance(items[0], dict):
        return items[0].get(key)
    return None


# ══════════════════════════════════════════════════════════════════
# HELPER PDF
# ══════════════════════════════════════════════════════════════════

def _extraire_page(page) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return page.extract_text() or ""


async def _telecharger_pdf(pdf_url: str, zone: str, max_chars: int = 15000) -> dict:
    """Télécharge le PDF règlement PLU (HTTP/2) et extrait le texte de la zone."""
    async with httpx.AsyncClient(
        timeout=60, follow_redirects=True, http2=True, headers=PDF_HEADERS
    ) as client:
        r = await client.get(pdf_url)

    if r.status_code != 200 or len(r.content) < 1000:
        return {"succes": False, "erreur": f"HTTP {r.status_code}", "texte": "", "nb_pages": 0, "pages_zone": []}

    try:
        reader    = PdfReader(io.BytesIO(r.content))
        nb_pages  = len(reader.pages)
        zu        = zone.upper()
        zone_pgs  = []
        gen_pgs   = []

        for i, page in enumerate(reader.pages):
            t  = _extraire_page(page)
            tu = t.upper()
            if any([
                f"ZONE {zu}" in tu, f"ARTICLE {zu}" in tu, f"SECTION {zu}" in tu,
                f"{zu}.1" in tu, f"{zu}.2" in tu, f"{zu}.3" in tu,
                f"\n{zu} -" in tu, f"\n{zu}\n" in tu,
            ]):
                zone_pgs.append((i, t))
            elif i < 20 and any(kw in tu for kw in ["DISPOSITION GÉNÉRALE", "DISPOSITION GENERALE", "SOMMAIRE", "DÉFINITION"]):
                gen_pgs.append((i, t))

        # Page suivante de chaque page zone (continuité des articles)
        zone_idx = {p[0] for p in zone_pgs}
        ctx_pgs  = [(i+1, _extraire_page(reader.pages[i+1])) for i, _ in zone_pgs if i+1 < nb_pages and i+1 not in zone_idx]

        combined = ""
        for _, t in gen_pgs[:2]:
            combined += t + "\n"
        for _, t in zone_pgs + ctx_pgs:
            combined += t + "\n"
            if len(combined) > max_chars:
                break

        if not zone_pgs:
            combined = ""
            for i in range(min(20, nb_pages)):
                combined += _extraire_page(reader.pages[i]) + "\n"
                if len(combined) > max_chars:
                    break

        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n[... tronqué ...]"

        return {
            "succes": True, "texte": combined,
            "nb_pages": nb_pages, "pages_zone": [p[0] for p in zone_pgs],
            "taille_ko": len(r.content) // 1024,
        }
    except Exception as e:
        return {"succes": False, "erreur": str(e), "texte": "", "nb_pages": 0, "pages_zone": []}


# ══════════════════════════════════════════════════════════════════
# CŒUR : ANALYSE TECHNIQUE (16 APIs en parallèle)
# ══════════════════════════════════════════════════════════════════

async def _run_analyse_technique(idu: str, lon: float, lat: float, code_insee: str, geom: dict) -> dict:
    """
    Lance 16 appels API en parallèle via asyncio.gather.
    Retourne un dict structuré : sol / risques / servitudes / environnement / acces / commune.
    Les 3 endpoints Géorisques en 503 (argiles, radon, AZI) ont un retry automatique.
    """
    g  = urllib.parse.quote(json.dumps(geom))
    ll = f"{lon},{lat}"

    (
        gaspar, sismicite, cavites, mvt, icpe,
        argiles, radon, azi,
        sup_s, sup_l, prescriptions,
        natura, znieff1, znieff2,
        voirie, commune,
    ) = await asyncio.gather(
        # ── Géorisques ───────────────────────────────────────────
        _safe_get(f"{GR}/gaspar/risques?code_insee={code_insee}"),
        _safe_get(f"{GR}/zonage_sismique?code_insee={code_insee}"),
        _safe_get(f"{GR}/cavites?latlon={ll}&rayon=1000"),
        _safe_get(f"{GR}/mvt?latlon={ll}&rayon=1000"),
        _safe_get(f"{GR}/installations_classees?latlon={ll}&rayon=500"),
        _safe_get(f"{GR}/argiles?lon={lon}&lat={lat}"),          # 503 → retry
        _safe_get(f"{GR}/radon?code_commune={code_insee}"),      # 503 → retry
        _safe_get(f"{GR}/azi?lon={lon}&lat={lat}"),              # 503 → retry
        # ── GPU servitudes ────────────────────────────────────────
        _safe_get(f"{AC}/gpu/assiette-sup-s?geom={g}"),
        _safe_get(f"{AC}/gpu/assiette-sup-l?geom={g}"),
        _safe_get(f"{AC}/gpu/prescription-surf?geom={g}"),
        # ── Environnement ─────────────────────────────────────────
        _safe_get(f"{AC}/nature/natura-habitat?geom={g}"),
        _safe_get(f"{AC}/nature/znieff1?geom={g}"),
        _safe_get(f"{AC}/nature/znieff2?geom={g}"),
        # ── Voirie & commune ──────────────────────────────────────
        _safe_get(f"{BAN}/reverse/?lon={lon}&lat={lat}"),
        _safe_get(f"{GEO}/communes/{code_insee}?fields=nom,population,surface"),
        return_exceptions=True,
    )

    # ── Parser les SUP par type ───────────────────────────────────
    def sup_filter(r, prefixes):
        return [
            {"suptype": f["properties"].get("suptype"), "nom": f["properties"].get("nomsuplitt", "")}
            for f in _features(r)
            if f.get("properties", {}).get("suptype", "")[:2].lower() in prefixes
        ]

    abf       = sup_filter(sup_s, {"ac"})   # AC1, AC4 = Abords MH / Secteur sauvegardé
    carrieres = sup_filter(sup_s, {"pm"})   # PM1 = Risques miniers / anciennes carrières
    reseaux   = sup_filter(sup_l, {"el", "i4", "pt"})  # EL1/EL7 = HT, I4 = pipelines, PT1 = télécoms

    prescriptions_list = [
        f["properties"].get("libelle", "")
        for f in _features(prescriptions)
        if f.get("properties", {}).get("libelle")
    ]

    # ── Voirie ───────────────────────────────────────────────────
    voirie_feat  = _features(voirie)
    voirie_props = voirie_feat[0]["properties"] if voirie_feat else {}

    # ── Risques naturels ─────────────────────────────────────────
    risques_liste = _first(gaspar, "risques_detail") or []
    risques_noms  = [r.get("libelle_risque_long", "") for r in risques_liste if isinstance(r, dict)]

    # ── Résultat structuré ────────────────────────────────────────
    return {
        "sol": {
            "alea_argiles":   _first(argiles, "alea_rga") or _first(argiles, "alea"),
            "radon_categorie": _first(radon, "categorie_potentiel_radon"),
            "cavites":        [{"type": d.get("type_cavite"), "id": d.get("identifiant")} for d in _data(cavites)],
            "mvt_terrain":    [{"type": d.get("type"), "date": d.get("date_mouvement")} for d in _data(mvt)],
            "nb_cavites":     len(_data(cavites)),
            "nb_mvt":         len(_data(mvt)),
        },
        "risques": {
            "liste_naturels":  risques_noms,
            "sismicite":       _first(sismicite, "zone_sismicite"),
            "code_sismique":   _first(sismicite, "code_zone"),
            "inondabilite":    _data(azi),
            "zone_inondable":  bool(_data(azi)),
            "icpe_nb":         len(_data(icpe)),
            "icpe_proches":    [{"nom": d.get("raisonSociale"), "adresse": d.get("adresse1")} for d in _data(icpe)[:5]],
        },
        "servitudes": {
            "abf":              abf,
            "nb_abf":           len(abf),
            "carrieres":        carrieres,
            "nb_carrieres":     len(carrieres),
            "reseaux":          reseaux,
            "nb_reseaux":       len(reseaux),
            "prescriptions_plu": prescriptions_list,
            "nb_prescriptions": len(prescriptions_list),
        },
        "environnement": {
            "natura2000":   [f["properties"].get("nom_site", "") for f in _features(natura)],
            "znieff":       [f["properties"].get("nom_znieff", "") for f in (_features(znieff1) + _features(znieff2))],
            "nb_natura":    len(_features(natura)),
            "nb_znieff":    len(_features(znieff1)) + len(_features(znieff2)),
        },
        "acces": {
            "voie":      voirie_props.get("street"),
            "type_voie": voirie_props.get("type"),
            "adresse":   voirie_props.get("label"),
            "distance_m": voirie_props.get("distance"),
        },
        "commune": commune if isinstance(commune, dict) else {},
    }


# ══════════════════════════════════════════════════════════════════
# ROUTE 1 : PARCELLE
# ══════════════════════════════════════════════════════════════════

@app.post("/parcelle")
async def get_parcelle(req: ParcelleRequest):
    params = {
        "code_insee": req.code_insee,
        "section":    req.section.upper(),
        "numero":     req.numero.zfill(4),
        "com_abs":    req.com_abs.zfill(3),
    }
    cad_url = f"https://apicarto.ign.fr/api/cadastre/parcelle?" + urllib.parse.urlencode(params)

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(cad_url)
        r.raise_for_status()
        cad_data = r.json()

    features = cad_data.get("features", [])
    if not features:
        raise HTTPException(status_code=404, detail="Parcelle introuvable.")

    if len(features) > 1:
        return {
            "status": "ambigue",
            "message": f"{len(features)} parcelles trouvées. Précisez com_abs.",
            "choix": [
                {"idu": f["properties"]["idu"], "surface": f["properties"]["contenance"],
                 "commune": f["properties"]["nom_com"], "com_abs": f["properties"]["com_abs"],
                 "section": f["properties"]["section"], "numero": f["properties"]["numero"]}
                for f in features
            ],
        }

    props = features[0]["properties"]
    geom  = features[0]["geometry"]
    idu   = props["idu"]

    # GPU zone PLU + URL PDF
    geom_enc = urllib.parse.quote(json.dumps(geom))
    zone_plu = type_zone = libelle_zone = pdf_url = nom_document = type_document = idurba = None

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r_zone = await client.get(f"https://apicarto.ign.fr/api/gpu/zone-urba?geom={geom_enc}")
            if r_zone.status_code == 200:
                zones = r_zone.json().get("features", [])
                if zones:
                    z             = zones[0]["properties"]
                    zone_plu      = z.get("libelle")
                    type_zone     = z.get("typezone")
                    libelle_zone  = z.get("libelong", "")
                    idurba        = z.get("idurba")
                    nomfic        = z.get("nomfic", "")
                    urlfic        = z.get("urlfic", "")
                    partition     = z.get("partition", "")

                    r_doc = await client.get(f"https://apicarto.ign.fr/api/gpu/document?geom={geom_enc}")
                    if r_doc.status_code == 200:
                        docs = r_doc.json().get("features", [])
                        if docs:
                            d             = docs[0]["properties"]
                            nom_document  = d.get("grid_title")
                            type_document = d.get("du_type")
                            gpu_doc_id    = d.get("gpu_doc_id", "")
                            pdf_url = urlfic or (
                                f"https://data.geopf.fr/annexes/gpu/documents/{partition}/{gpu_doc_id}/{nomfic}"
                                if gpu_doc_id and partition and nomfic else None
                            )
        except Exception:
            pass

    return {
        "status": "ok",
        "idu": idu,
        "surface_parcelle": props["contenance"],
        "commune":          props["nom_com"],
        "code_commune":     props["code_insee"],
        "section":          props["section"],
        "numero":           props["numero"],
        "com_abs":          props["com_abs"],
        "code_arr":         props.get("code_arr"),
        "geometrie":        geom,
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


# ══════════════════════════════════════════════════════════════════
# ROUTE 2 : BÂTIMENT (BDNB)
# ══════════════════════════════════════════════════════════════════

@app.post("/batiment")
async def get_batiment(req: BatimentRequest):
    async with httpx.AsyncClient(timeout=20) as client:
        r_rel = await client.get(
            f"{BDNB}/rel_batiment_groupe_parcelle?parcelle_id=eq.{req.idu}&select=batiment_groupe_id&limit=5",
            headers={"Accept": "application/json"},
        )
        if r_rel.status_code != 200 or not r_rel.json():
            return {"status": "not_found", "message": "Aucun bâtiment BDNB trouvé.", "idu": req.idu}

        bat_id = r_rel.json()[0]["batiment_groupe_id"]

        r_bat = await client.get(
            f"{BDNB}/batiment_groupe_complet"
            f"?batiment_groupe_id=eq.{bat_id}"
            f"&select=batiment_groupe_id,s_geom_groupe,nb_niveau,annee_construction,"
            f"usage_niveau_1_txt,mat_mur_txt,mat_toit_txt,nb_log,"
            f"libelle_adr_principale_ban,classe_bilan_dpe&limit=1",
            headers={"Accept": "application/json"},
        )
        if r_bat.status_code != 200 or not r_bat.json():
            return {"status": "partial", "batiment_groupe_id": bat_id}

        b = r_bat.json()[0]
        return {
            "status":             "ok",
            "idu":                req.idu,
            "batiment_groupe_id": bat_id,
            "surface_au_sol":     b.get("s_geom_groupe"),
            "nb_niveaux":         b.get("nb_niveau"),
            "annee_construction": b.get("annee_construction"),
            "usage":              b.get("usage_niveau_1_txt"),
            "materiaux_mur":      b.get("mat_mur_txt"),
            "materiaux_toit":     b.get("mat_toit_txt"),
            "nb_logements":       b.get("nb_log"),
            "adresse":            b.get("libelle_adr_principale_ban"),
            "classe_dpe":         b.get("classe_bilan_dpe"),
        }


# ══════════════════════════════════════════════════════════════════
# ROUTE 3 : ANALYSE TECHNIQUE (16 APIs)
# ══════════════════════════════════════════════════════════════════

@app.post("/technique")
async def get_technique(req: TechniqueRequest):
    """
    Lance 16 APIs en parallèle et retourne l'analyse technique complète :
    sol, risques, servitudes, environnement, accès, commune.
    """
    result = await _run_analyse_technique(
        idu        = req.idu,
        lon        = req.lon,
        lat        = req.lat,
        code_insee = req.code_insee,
        geom       = req.geom,
    )
    return {"status": "ok", "idu": req.idu, **result}


# ══════════════════════════════════════════════════════════════════
# ROUTE 4 : ANALYSE CLAUDE (PLU + technique → JSON structuré)
# ══════════════════════════════════════════════════════════════════

@app.post("/analyse")
async def analyse_plu(req: AnalyseRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Clé Anthropic manquante.")

    # ── A : PDF règlement ─────────────────────────────────────────
    pdf_result = await _telecharger_pdf(req.pdf_url, req.zone_plu)

    if pdf_result["succes"]:
        bloc_pdf = (
            f"=== RÈGLEMENT PLU (PDF téléchargé — {pdf_result['taille_ko']} Ko, "
            f"{pdf_result['nb_pages']} pages, pages zone : {pdf_result['pages_zone']}) ===\n"
            f"{pdf_result['texte']}\n=== FIN RÈGLEMENT ==="
        )
    else:
        bloc_pdf = (
            f"PDF non accessible ({pdf_result.get('erreur')}). "
            f"Base-toi sur tes connaissances générales de la zone {req.zone_plu} "
            f"et précise que l'analyse est générique."
        )

    # ── B : bâtiment existant ─────────────────────────────────────
    bat = req.batiment
    if bat and bat.get("status") == "ok":
        bloc_batiment = (
            f"BÂTIMENT EXISTANT (BDNB) :\n"
            f"- Surface au sol : {bat.get('surface_au_sol')} m²\n"
            f"- Niveaux : {bat.get('nb_niveaux')}\n"
            f"- Année : {bat.get('annee_construction')}\n"
            f"- Usage : {bat.get('usage')}\n"
            f"- Matériaux murs : {bat.get('materiaux_mur')}\n"
            f"- DPE : {bat.get('classe_dpe')} | Logements : {bat.get('nb_logements')}"
        )
    else:
        bloc_batiment = "Aucun bâtiment existant BDNB (parcelle non bâtie ou données indisponibles)."

    # ── C : données techniques ────────────────────────────────────
    tech = req.technique
    if tech:
        sol  = tech.get("sol", {})
        risq = tech.get("risques", {})
        serv = tech.get("servitudes", {})
        envi = tech.get("environnement", {})
        accs = tech.get("acces", {})
        com  = tech.get("commune", {})

        # Construire les alertes critiques
        alertes = []
        if risq.get("zone_inondable"):
            alertes.append("🚨 ZONE INONDABLE détectée — peut bloquer le permis")
        if serv.get("nb_abf", 0) > 0:
            abf_noms = ", ".join(s.get("nom", "") for s in serv.get("abf", []))
            alertes.append(f"⚠ PÉRIMÈTRE ABF : {abf_noms} → +3 à 6 mois délai PC, contraintes esthétiques")
        if serv.get("nb_carrieres", 0) > 0:
            alertes.append(f"⚠ RISQUE MINIER/CARRIÈRES (PM1) → étude géotechnique obligatoire")
        if sol.get("nb_cavites", 0) > 0:
            alertes.append(f"⚠ {sol['nb_cavites']} CAVITÉ(S) souterraine(s) à proximité")
        if sol.get("nb_mvt", 0) > 0:
            alertes.append(f"⚠ {sol['nb_mvt']} mouvement(s) de terrain historique(s)")
        if sol.get("alea_argiles") and sol["alea_argiles"].upper() in ("FORT", "TRÈS FORT", "MOYEN"):
            alertes.append(f"⚠ ARGILES {sol['alea_argiles']} → fondations renforcées (+5k€ à +50k€)")
        if serv.get("nb_reseaux", 0) > 0:
            alertes.append(f"⚠ {serv['nb_reseaux']} réseau(x) souterrain(s)/aérien(s) (HT, gaz, télécoms)")
        if envi.get("nb_natura", 0) > 0:
            alertes.append(f"⚠ NATURA 2000 → Évaluation des Incidences obligatoire")

        bloc_technique = f"""
ANALYSE TECHNIQUE (données réelles — APIs Géorisques + GPU + IGN) :

SOL & GÉOLOGIE :
- Aléa argiles (retrait-gonflement) : {sol.get('alea_argiles') or 'Non disponible'}
- Potentiel radon : {sol.get('radon_categorie') or 'Non disponible'}
- Cavités souterraines dans un rayon 1km : {sol.get('nb_cavites', 0)}
- Mouvements de terrain historiques : {sol.get('nb_mvt', 0)}

RISQUES NATURELS & INDUSTRIELS :
- Risques officiels commune (GASPAR) : {', '.join(risq.get('liste_naturels', [])) or 'Aucun'}
- Zone sismique : {risq.get('sismicite') or 'Non disponible'}
- Zone inondable (AZI/PPRI) : {'OUI ⚠' if risq.get('zone_inondable') else 'Non détectée'}
- Installations classées ICPE dans 500m : {risq.get('icpe_nb', 0)}

SERVITUDES D'UTILITÉ PUBLIQUE (GPU) :
- Périmètres ABF (monuments historiques) : {serv.get('nb_abf', 0)} trouvé(s)
  {chr(10).join(f"  • {s.get('suptype','').upper()} — {s.get('nom','')}" for s in serv.get('abf',[]))}
- Risques miniers / carrières : {serv.get('nb_carrieres', 0)} trouvé(s)
  {chr(10).join(f"  • {s.get('suptype','').upper()} — {s.get('nom','')}" for s in serv.get('carrieres',[]))}
- Réseaux (HT, pipelines, télécoms) : {serv.get('nb_reseaux', 0)} trouvé(s)
  {chr(10).join(f"  • {s.get('suptype','').upper()} — {s.get('nom','')}" for s in serv.get('reseaux',[]))}
- Prescriptions PLU localisées : {serv.get('nb_prescriptions', 0)}
  {chr(10).join(f"  • {p}" for p in serv.get('prescriptions_plu',[]))}

ENVIRONNEMENT :
- Natura 2000 : {envi.get('nb_natura', 0)} site(s)
- ZNIEFF : {envi.get('nb_znieff', 0)} zone(s)

ACCESSIBILITÉ :
- Voie d'accès : {accs.get('voie') or 'Non identifiée'}
- Type de voie : {accs.get('type_voie') or '-'}
- Distance voie : {accs.get('distance_m') or '-'} m

COMMUNE :
- Nom : {com.get('nom') or '-'} | Population : {com.get('population') or '-'} hab | Surface : {com.get('surface') or '-'} ha

ALERTES CRITIQUES IDENTIFIÉES :
{chr(10).join(alertes) if alertes else "✓ Aucune alerte critique détectée"}
"""
    else:
        bloc_technique = "Données techniques non disponibles pour cette analyse."

    # ── D : prompt Claude ─────────────────────────────────────────
    prompt = f"""Tu es un expert en droit de l'urbanisme français et en promotion immobilière.

PARCELLE ANALYSÉE :
- IDU : {req.idu}
- Commune : {req.commune}
- Surface : {req.surface_parcelle} m²
- Zone PLU : {req.zone_plu}

{bloc_batiment}

{bloc_technique}

{bloc_pdf}

Réponds UNIQUEMENT en JSON valide, sans texte avant ni après :

{{
  "zone": "{req.zone_plu}",
  "type_zone": "U/AU/A/N",
  "destination_principale": "description courte",
  "ce_qui_est_autorise": ["usage 1", "usage 2"],
  "ce_qui_est_interdit": ["interdit 1", "interdit 2"],
  "regles_hauteur": {{
    "hauteur_max": "valeur avec unité ou null",
    "regles_specifiques": "description précise avec article si disponible"
  }},
  "regles_implantation": {{
    "retrait_rue": "valeur ou règle",
    "retrait_limites": "valeur ou règle",
    "emprise_sol_max": "% ou null"
  }},
  "surface_constructible": {{
    "shon_estimee": "estimation en m² pour cette parcelle",
    "nb_niveaux_max": "nombre ou null",
    "methode_calcul": "explication"
  }},
  "stationnement": "règles applicables",
  "performances_energetiques": "RE2020 ou autre",
  "contraintes_techniques": {{
    "fondations": "type recommandé selon aléa argiles et cavités",
    "delai_pc_estime": "estimation en mois avec justification (ABF, standard)",
    "surcoût_estime": "estimation surcoût lié aux contraintes techniques",
    "points_bloquants": ["contrainte 1", "contrainte 2"]
  }},
  "points_attention": ["point 1", "point 2", "point 3"],
  "opportunite_promoteur": {{
    "score": 65,
    "commentaire": "évaluation synthétique tenant compte des contraintes techniques"
  }},
  "source_analysee": "{req.pdf_url}"
}}
"""

    client_ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        message = client_ai.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2500,
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
        return {"status": "parse_error", "message": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Claude : {e}")


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agent Foncier API", "version": "3.0"}
