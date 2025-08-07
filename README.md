# Carmila — Exécution locale

## Prérequis
- Python 3.11 recommandé
- Fichiers d'entrée dans `./data` :
  - `Transco.xlsx`
  - `2023 flux quotidiens.xlsx`
  - `2024 flux quotidiens.xlsx`
  - *(optionnel)* `Recensement OE 2024 - document de référence.xlsx`

## Installation
```bash
python -m venv .venv
# Windows
. .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer
- **Sans Drive / GCS (local)** :
  ```bash
  python main.py
  ```
- **Avec Drive + GCS** :
  ```bash
  # Windows (PowerShell)
  set DRIVE_FOLDER_ID=1jjvQJUQMKCqZ5KbS0NyBMdJh7Zl-ublt
  set GCS_BUCKET_NAME=carmila-end-bucket
  python main.py

  # macOS / Linux (bash)
  export DRIVE_FOLDER_ID=1jjvQJUQMKCqZ5KbS0NyBMdJh7Zl-ublt
  export GCS_BUCKET_NAME=carmila-end-bucket
  python main.py
  ```

> En cloud (Cloud Run Job), seul `/tmp` est en écriture. Le script utilise automatiquement `/tmp/data` et `/tmp/tmp_data`.

## Sorties
- `tmp_data/carmila_pred_2025.csv` (UTF-8-SIG, séparateur `;`)
- `data/Carmila - Alberthon - Time Series - Data_new.csv`

## Schéma du CSV final
- `nom_mall` *(str)*
- `cluster` *(str)*
- `date` *(YYYY-MM-DD)*
- `donnees_reelles` *(int ou vide)*
- `donnees_predites` *(int ou vide)*

## Dépannage rapide (local)
- **Fichier manquant** → vérifier les noms exacts dans `./data` (sensibles à la casse).
- **openpyxl** manquant → `pip install openpyxl`.
- **Mémoire** : si vous activez des libs lourdes (torch, darts), pensez à fermer les autres apps.

## Contact / Handover
- Projet GCP : `carmila-end` (région `europe-west9`)
- Bucket sortie : `gs://carmila-end-bucket/carmila/carmila_pred_2025.csv`
- Dossier Drive source : `1jjvQJUQMKCqZ5KbS0NyBMdJh7Zl-ublt`
```
