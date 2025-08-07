#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — pipeline unifiée (stratégie d’inférence hebdo + prédire tous les jours 2025)
- Entraîne un modèle unique par mall (un seul fit)
- Entraînement sur tous les réels disponibles (inclut 2025 si présents)
- Prédit seulement les jours 2025 sans données réelles (les anciennes prédictions sont préservées)
- Jours fermés -> prédiction = 0
- Inférence hebdomadaire (mise à jour Lag7 semaine -> semaine suivante)
- Périmètre de flux : filtre "Zone = centre" si colonne zone présente
- Export cumulatif : nom_mall;cluster;date;donnees_reelles;donnees_predites
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# — Constantes globales --------------------------------------------------------
OUTPUT_YEAR = 2025
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# — Env / chemins (Cloud Run-friendly) ----------------------------------------
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

BASE_DIR = Path(os.getenv("WORKDIR", "/tmp"))
TMP = BASE_DIR / "tmp_data";  TMP.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"; DATA_DIR.mkdir(exist_ok=True)

RAW_TS = DATA_DIR / "Carmila - Alberthon - Time Series - Data_new.csv"
OUT_CSV = TMP / "carmila_pred_2025.csv"

# — Logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# — Helpers divers -------------------------------------------------------------
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s

canon = lambda s: nfc(str(s)).strip().upper()

# -------------------------- Google Drive (optionnel) --------------------------
if DRIVE_FOLDER_ID:
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    def drive_service():
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def download_drive(folder_id: str) -> List[Path]:
        svc = drive_service()
        q = f"'{folder_id}' in parents and trashed=false"
        files, token = [], None
        log.info("Listing du dossier Drive…")
        while True:
            resp = svc.files().list(
                q=q,
                fields="nextPageToken, files(id,name,createdTime)",
                pageToken=token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            files.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        if not files:
            log.error("Dossier Drive vide.")
            return []

        files.sort(key=lambda x: x.get("createdTime", ""))
        dled = []
        for f in files:
            dest = TMP / f["name"]
            log.info("↓ %s", f["name"])
            req = svc.files().get_media(fileId=f["id"], supportsAllDrives=True)
            with io.FileIO(dest, "wb") as fh:
                dlr = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = dlr.next_chunk()
            dled.append(dest)
        return dled
else:
    def download_drive(_: str) -> List[Path]:
        log.debug("Download Drive désactivé (DRIVE_FOLDER_ID absent).")
        return []

# --------------------------- Google Cloud Storage -----------------------------
if GCS_BUCKET_NAME:
    from google.cloud import storage

    def upload_to_gcs(bucket: str, local: Path, remote: str):
        storage.Client().bucket(bucket).blob(remote).upload_from_filename(local)
        log.info("Upload → gs://%s/%s", bucket, remote)
else:
    def upload_to_gcs(_: str, __: Path, ___: str):
        log.debug("Upload GCS désactivé (GCS_BUCKET_NAME absent).")

# — Feature engineering commun -------------------------------------------------
def make_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["Month name"] = pd.Categorical(df["Date"].dt.month_name(), categories=MONTHS, ordered=True)
    df["Week day name"] = pd.Categorical(df["Date"].dt.day_name(), categories=WEEKDAYS, ordered=True)
    df["Consumption events"] = 0
    df["Consumption events LW"] = df.groupby("ID mall")["Consumption events"].shift(7).fillna(0).astype(int)
    return df

# — Parsing recensement (jours exceptionnels) ----------------------------------
def parse_recensement(rec_path: Path) -> Dict:
    rec = pd.read_excel(rec_path, sheet_name="RECAP OE 2024", header=1)
    rec = rec[rec["CENTRE"].notna()].copy()
    rec["mall_name"] = rec["CENTRE"].map(canon)
    melted = rec.melt(id_vars="mall_name", var_name="DateStr", value_name="Status")

    def map_status(v):
        s = str(v).strip().lower()
        if s in {"true", "1", "1.0"}:
            return 1
        if s in {"false", "0", "0.0"}:
            return -1
        return np.nan

    MONTH_FR = {
        "janvier": "January", "février": "February", "fevrier": "February",
        "mars": "March", "avril": "April", "mai": "May", "juin": "June",
        "juillet": "July", "août": "August", "aout": "August",
        "septembre": "September", "octobre": "October", "novembre": "November",
        "décembre": "December", "decembre": "December"
    }

    def fr_date(txt):
        m = re.search(r"(\d{1,2})\s+([a-zéû]+)", str(txt).lower().replace("1er", "1"))
        if not m:
            return pd.NaT
        d, mfr = m.groups()
        return pd.to_datetime(f"{d} {MONTH_FR.get(mfr,'')} 2024", format="%d %B %Y", errors="coerce")

    melted["Exceptional"] = melted["Status"].apply(map_status)
    melted["Date"] = melted["DateStr"].apply(fr_date)
    melted.dropna(subset=["Date", "Exceptional"], inplace=True)
    melted["MonthDay"] = melted["Date"].dt.strftime("%m-%d")
    return melted.set_index(["mall_name", "MonthDay"])["Exceptional"].to_dict()

# — Création du master dataset -------------------------------------------------
def build_master(transco_p: Path, flux_ps: List[Path], rec_p: Optional[Path]) -> None:
    tr = (pd.read_excel(transco_p) if transco_p.suffix.lower() != ".csv" else pd.read_csv(transco_p, sep=","))
    required = {"ID mall", "Site", "Cluster"}
    if missing := required.difference(tr.columns):
        log.error("Transco sans colonnes %s", ", ".join(missing))
        sys.exit(1)
    tr = tr[tr["ID mall"] != "-"].dropna(subset=["ID mall"]).copy()
    tr["mall_name"] = tr["Site"].map(canon)
    tr["ID mall"] = tr["ID mall"].astype(int)

    all_flux = []
    for p in flux_ps:
        if not p.exists():
            continue
        # lecture
        df = (pd.read_excel(p) if p.suffix.lower() != ".csv" else pd.read_csv(p, sep=",", encoding="utf-8", errors="replace"))
        # normalisation colonnes
        colmap = {"Centre": "Site", "Mall": "Site", "Jour": "Date", "Date": "Date", "Entrées": "Real", "Entrees": "Real"}
        df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}, inplace=True)

        # Filtre "Zone = centre" si une colonne zone existe
        zone_col = next((c for c in df.columns if re.search(r"zone", c, re.I)), None)
        if zone_col:
            df = df[df[zone_col].astype(str).str.contains("centre", case=False, na=False)]

        df["Site"] = df["Site"].map(canon)
        df = df.merge(tr[["mall_name", "ID mall"]], left_on="Site", right_on="mall_name", how="left")
        df.dropna(subset=["ID mall", "Date", "Real"], inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Real"] = pd.to_numeric(df["Real"], errors="coerce")
        df.dropna(subset=["Real"], inplace=True)
        df["Real"] = df["Real"].astype(int)
        all_flux.append(df[["ID mall", "Date", "Real"]])

    if not all_flux:
        log.error("Aucun flux exploitable.")
        sys.exit(1)

    hist = pd.concat(all_flux, ignore_index=True).drop_duplicates(["ID mall", "Date"], keep="last")
    first_date = hist["Date"].min()
    malls_with_data = hist["ID mall"].unique()

    full = (
        pd.MultiIndex
        .from_product([malls_with_data, pd.date_range(first_date, f"{OUTPUT_YEAR}-12-31", freq="D")],
                      names=["ID mall", "Date"])
        .to_frame(index=False)
        .merge(hist.rename(columns={"Real": "Daily footfall"}), on=["ID mall", "Date"], how="left")
    )

    full["Daily footfall"] = full["Daily footfall"].astype("float")

    # Jours exceptionnels
    full["Exceptionnally Closed"] = 0
    full["Exceptionnally Open"] = 0
    full["Exceptionnal Hours"] = 0

    if rec_p and rec_p.exists():
        mapping = parse_recensement(rec_p)
        full["MonthDay"] = full["Date"].dt.strftime("%m-%d")
        mn = full.merge(tr[["ID mall", "mall_name"]], on="ID mall", how="left")
        exc = [mapping.get((m, d), 0) for m, d in zip(mn["mall_name"], full["MonthDay"])]
        full["Exceptionnally Closed"] = [1 if e == -1 else 0 for e in exc]
        full["Exceptionnally Open"] = [1 if e == 1 else 0 for e in exc]
        full.drop(columns="MonthDay", inplace=True)
    else:
        log.warning("Fichier recensement OE absent.")

    full.sort_values(["ID mall", "Date"], inplace=True)
    full = make_feature_cols(full)
    full.to_csv(RAW_TS, index=False)
    log.info("Master Time Series reconstruit → %s (%d lignes)", RAW_TS, len(full))

# ====== Modèle (fit unique/mall) + inférence hebdo ============================
FEATURES_DOCKER = ["WeekDay", "Month", "WeekOfYear", "Lag7", "ExceptionalDayStatus"]

def _prepare_features_docker(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Daily footfall" in d.columns:
        d = d.rename(columns={"Daily footfall": "Real"})

    # ExceptionalDayStatus = Open(1) - Closed(1)
    if "ExceptionalDayStatus" not in d.columns:
        open_col, close_col = "Exceptionnally Open", "Exceptionnally Closed"
        d[open_col], d[close_col] = d.get(open_col, 0), d.get(close_col, 0)
        d["ExceptionalDayStatus"] = d[open_col].fillna(0).astype(int) - d[close_col].fillna(0).astype(int)

    # Historique : si jour fermé et valeur présente, on force Real=0 pour l'apprentissage
    mask_closed_hist = (d["ExceptionalDayStatus"] < 0) & (d["Real"].notna())
    d.loc[mask_closed_hist, "Real"] = 0.0

    d = d.sort_values(["ID mall", "Date"])
    d["WeekDay"]    = d["Date"].dt.weekday
    d["Month"]      = d["Date"].dt.month
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["Lag7"]       = d.groupby("ID mall")["Real"].shift(7)
    return d

def _predict_per_mall_rolling_into_master(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Inférence par semaines (hebdo) :
    - Fit unique par mall sur tous les réels disponibles
    - On ne prédit QUE les jours 2025 encore sans réel
    - Les anciennes prédictions ne sont donc jamais ré-écrasées
    """
    d = _prepare_features_docker(df_master)
    preds_all = []

    for mall_id, g in d.groupby("ID mall"):
        g = g.copy()

        # Un seul fit par mall
        train_mask = g["Real"].notna()
        if train_mask.sum() < 20:
            log.warning("Mall %s : série trop courte (%d pts), ignoré.", mall_id, train_mask.sum())
            continue

        model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        X_train = g.loc[train_mask, FEATURES_DOCKER].fillna(0)
        y_train = g.loc[train_mask, "Real"]
        model.fit(X_train, y_train)

        # Fenêtrage hebdo: du lundi 2024-12-30 au 2025-12-29
        week_start = pd.Timestamp("2024-12-30")
        last_week_start = pd.Timestamp(f"{OUTPUT_YEAR}-12-29")

        while week_start <= last_week_start:
            week_end = week_start + pd.Timedelta(days=6)

            mask_week = (g["Date"] >= week_start) & (g["Date"] <= week_end)

            # <-- NOUVEAU : prédire seulement les jours 2025 sans réel
            mask_week_2025_missing = (
                mask_week
                & (g["Date"].dt.year == OUTPUT_YEAR)
                & (g["Real"].isna())
            )

            if mask_week_2025_missing.any():
                Xw = g.loc[mask_week_2025_missing, FEATURES_DOCKER].fillna(0).copy()
                yhat = model.predict(Xw).astype(float)

                # Fermeture -> 0
                exc = g.loc[mask_week_2025_missing, "ExceptionalDayStatus"].to_numpy()
                yhat[exc < 0] = 0.0

                yhat = np.maximum(0, np.rint(yhat)).astype(int)

                tmp = g.loc[mask_week_2025_missing, ["ID mall", "Date"]].copy()
                tmp["donnees_predites"] = yhat
                preds_all.append(tmp)

            # Mise à jour de Lag7 semaine -> semaine suivante
            next_mask = (g["Date"] > week_end) & (g["Date"] <= week_end + pd.Timedelta(days=7))
            if next_mask.any():
                Xw_full = g.loc[mask_week, FEATURES_DOCKER].fillna(0)
                if len(Xw_full):
                    yhat_full = model.predict(Xw_full).astype(float)
                    yhat_full = np.maximum(0, np.rint(yhat_full)).astype(int)
                    k = int(next_mask.sum())
                    g.loc[next_mask, "Lag7"] = yhat_full[:k]

            week_start += pd.Timedelta(days=7)

    out = df_master.copy()
    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
        out = out.merge(preds_df, on=["ID mall", "Date"], how="left")
    else:
        out["donnees_predites"] = np.nan

    # Sécurité : pas de prédictions < 2025
    out.loc[out["Date"].dt.year < OUTPUT_YEAR, "donnees_predites"] = np.nan

    return out

# ====== Pipeline principal ====================================================
def pipeline() -> None:
    log.info("ENV → DRIVE_FOLDER_ID=%s, GCS_BUCKET_NAME=%s", DRIVE_FOLDER_ID or "(none)", GCS_BUCKET_NAME or "(none)")
    if DRIVE_FOLDER_ID:
        log.info("⇣ Téléchargement Drive…")
        download_drive(DRIVE_FOLDER_ID)
    else:
        log.info("Mode local : fichiers dans ./data")

    # Localisation des fichiers
    # NOUVEAU CODE AMÉLIORÉ
    transco_p = (TMP / "Transco.xlsx") if (TMP / "Transco.xlsx").exists() else (DATA_DIR / "Transco.xlsx")

    # --- Début de la logique de découverte améliorée ---
    log.info("Recherche des fichiers de flux...")

    # 1. On garde la liste des fichiers officiels
    flux_candidates = ["2023 flux quotidiens.xlsx", "2024 flux quotidiens.xlsx", "2025 flux quotidiens.xlsx"]
    all_potential_files: List[Path] = []

    for name in flux_candidates:
        # On cherche dans le dossier temporaire (Drive) puis dans le dossier local
        p_tmp = TMP / name
        p_data = DATA_DIR / name
        if p_tmp.exists():
            all_potential_files.append(p_tmp)
        elif p_data.exists():
            all_potential_files.append(p_data)

    # 2. EN PLUS, on cherche tout autre fichier contenant "flux" (xlsx ou csv)
    for folder in [TMP, DATA_DIR]:
        all_potential_files.extend(folder.glob("*flux*.xlsx"))
        all_potential_files.extend(folder.glob("*flux*.csv"))

    # 3. On s'assure que chaque fichier est unique pour éviter de lire les données en double
    if all_potential_files:
        # Crée un dictionnaire où les clés sont les noms de fichiers, éliminant les doublons
        unique_flux_files = {p.name: p for p in all_potential_files}
        flux_ps = list(unique_flux_files.values()) # On reconvertit en liste
        log.info("Fichiers de flux qui seront chargés : %s", [p.name for p in flux_ps])
    else:
        flux_ps = []
    # --- Fin de la logique de découverte ---

    rec_glob = list(TMP.glob("*OE*2024*")) or list(DATA_DIR.glob("*OE*2024*"))
    rec_p = rec_glob[0] if rec_glob else None


    if not transco_p.exists():
        log.error("Transco absent : %s", transco_p)
        sys.exit(1)
    if not flux_ps:
        log.error("Aucun fichier de flux trouvé.")
        sys.exit(1)

    # 1) Master
    build_master(transco_p, flux_ps, rec_p)

    # 2) Prédictions
    df0 = pd.read_csv(RAW_TS, parse_dates=["Date"])
    dfp = _predict_per_mall_rolling_into_master(df0)

    # 3) Join transco + mapping colonnes, puis export final 5 colonnes
    tr = (pd.read_excel(transco_p) if transco_p.suffix.lower() != ".csv" else pd.read_csv(transco_p, sep=","))
    tr = tr[tr["ID mall"] != "-"].dropna(subset=["ID mall"]).copy()
    tr["ID mall"] = tr["ID mall"].astype(int)

    dfp = dfp.merge(tr[["ID mall", "Site", "Cluster"]], on="ID mall", how="left")
    dfp.rename(columns={"Site": "nom_mall", "Cluster": "cluster"}, inplace=True)

    dfp["donnees_reelles"] = dfp["Daily footfall"]

    final = (
        dfp[["nom_mall", "cluster", "Date", "donnees_reelles", "donnees_predites"]]
        .rename(columns={"Date": "date"})
        .sort_values(["nom_mall", "date"])
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------------------
    #  CONSERVER LES PREDICTIONS DÉJÀ PUBLIÉES
    # -----------------------------------------------------------------------------
    # … juste avant la fusion dans pipeline() …

    # 0) Si on utilise GCS, rapatrier l'ancien export dans TMP
    if GCS_BUCKET_NAME:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"carmila/{OUT_CSV.name}")
        if blob.exists():
            blob.download_to_filename(OUT_CSV)
            log.info("↳ Anciennes prédictions récupérées depuis GCS → %s", OUT_CSV)

    # 1) Maintenant on peut vraiment fusionner l’historique
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV, sep=",", parse_dates=["date"])
        # Garde-fou doublons
        assert not old.duplicated(["nom_mall", "cluster", "date"]).any(), "Duplicates in historical file"
        # Assemblage
        final = (
            final
            .merge(
                old[["nom_mall", "cluster", "date", "donnees_predites"]],
                on=["nom_mall", "cluster", "date"],
                how="left",
                suffixes=("", "_old"),
            )
        )
        final["donnees_predites"] = final["donnees_predites"].combine_first(final["donnees_predites_old"])
        final.drop(columns=["donnees_predites_old"], inplace=True)

    # 2) Écriture finale du CSV cumulatif
    final.to_csv(OUT_CSV, index=False, sep=",", encoding="utf-8-sig")
    log.info("✅ Export local : %s (%d lignes)", OUT_CSV, len(final))

    # 3) (Re)upload sur GCS pour la prochaine exécution
    if GCS_BUCKET_NAME:
        upload_to_gcs(GCS_BUCKET_NAME, OUT_CSV, f"carmila/{OUT_CSV.name}")


# — Entry point ----------------------------------------------------------------
if __name__ == "__main__":
    try:
        pipeline()
    except Exception as exc:
        log.exception("Erreur fatale : %s", exc)
        sys.exit(1)
