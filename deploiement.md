# Guide de Déploiement – Pipeline de Prédiction Carmila

Ce document décrit la procédure complète pour déployer le pipeline de prédiction automatisé sur un projet Google Cloud Platform (GCP).

## Objectif
Déployer un pipeline 100% serverless qui :

- Se déclenche sur une base horaire pour vérifier les modifications de fichiers dans un dossier Google Drive.
- Si des modifications sont détectées, exécute un job de prédiction conteneurisé.
- Le job se ré-entraîne avec les nouvelles données, génère des prédictions, et sauvegarde le résultat dans un bucket Cloud Storage.
- Le résultat est automatiquement chargé dans une table BigQuery pour être consommé par Looker Studio.

## Prérequis
- Un projet Google Cloud avec la facturation activée.
- Les outils en ligne de commande **gcloud** installés et configurés sur le poste de l'administrateur.
- **Docker Desktop** (ou Docker Engine) installé sur le poste de l'administrateur.

---

## Étape 1 : Configuration Initiale du Projet GCP
> Toutes les commandes suivantes sont à exécuter depuis un terminal.

### 1.1 – Définir les variables du projet
```bash
export PROJECT_ID="<ID_DE_VOTRE_PROJET_GCP>"
export REGION_SCHEDULER="europe-west1" # Région pour Scheduler et Function
export REGION_JOB="europe-west9"      # Région pour le Cloud Run Job
```

### 1.2 – Activer les APIs nécessaires
Cette commande active tous les services Google Cloud requis pour le pipeline.
```bash
gcloud services enable   iam.googleapis.com   run.googleapis.com   artifactregistry.googleapis.com   cloudfunctions.googleapis.com   cloudscheduler.googleapis.com   drive.googleapis.com   storage.googleapis.com   cloudbuild.googleapis.com   bigquery.googleapis.com   --project=$PROJECT_ID
```

### 1.3 – Créer les comptes de service
Nous utilisons deux comptes de service pour respecter le principe de moindre privilège :

- **job-carmila** : l'exécuteur. Il lance le job de prédiction et accède aux données.
- **scheduler-carmila** : le déclencheur. Il lance la Cloud Function.

```bash
# Compte pour le Job et la Fonction principale
gcloud iam service-accounts create job-carmila   --display-name="Service Account for Prediction Job"   --project=$PROJECT_ID

# Compte pour le Scheduler
gcloud iam service-accounts create scheduler-carmila   --display-name="Service Account for Scheduler Trigger"   --project=$PROJECT_ID
```

### 1.4 – Attribuer les permissions IAM
Le compte **job-carmila** a besoin de permissions pour écrire dans BigQuery et exécuter le job.

```bash
export SA_JOB_EMAIL="job-carmila@$PROJECT_ID.iam.gserviceaccount.com"

# BigQuery (écriture + lancement de jobs)
gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$SA_JOB_EMAIL"   --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$SA_JOB_EMAIL"   --role="roles/bigquery.jobUser"

# Cloud Run Jobs (exécuter/administrer le job)
gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$SA_JOB_EMAIL"   --role="roles/run.admin"

# Accès au bucket de sortie (écriture d'objets)
# (vous pouvez restreindre au niveau du bucket une fois créé, voir étape 4.1)
gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$SA_JOB_EMAIL"   --role="roles/storage.objectAdmin"
```

> **Note** : le compte **scheduler-carmila** n'a besoin d'aucun rôle supplémentaire, hormis l'invocation de la Cloud Function (ajoutée à l'étape 4.4).

---

## Étape 2 : Préparation de la Source (Google Drive)

### 2.1 – Créer un dossier Google Drive
Créez un dossier dans Google Drive qui contiendra tous les fichiers de données. Notez son **ID** à partir de l'URL (ex : `https://drive.google.com/drive/folders/<ID_DU_DOSSIER>`).

### 2.2 – Partager le dossier
Partagez ce dossier avec le compte de service **job-carmila** :

- Dans Google Drive, clic droit sur le dossier → **Partager**.
- Ajoutez l’adresse : `job-carmila@<ID_DE_VOTRE_PROJET_GCP>.iam.gserviceaccount.com`.
- Donnez le rôle **Lecteur** (Viewer).

### 2.3 – Conventions de nommage des fichiers
- **Fichiers de flux** : `2023 flux quotidiens.xlsx`, `2024 flux quotidiens.xlsx`, `2025 flux quotidiens.xlsx`, etc.
- **Fichier de transco** : `Transco.xlsx`
- **Fichier de recensement** : le nom doit contenir `OE` et `2024`.

---

## Étape 3 : Préparation du Code Source
Créez deux dossiers sur votre ordinateur :

- `carmila-prediction-job`
- `carmila-trigger-function`

### 3.1 – Code du Job de Prédiction (dossier : `carmila-prediction-job`)

**main.py** (le script avec apprentissage continu)  
> Collez ici le **dernier script `main.py` validé** (version avec ré-entraÎnement hebdo, verrou “fermé=0”, et réinjection des réels 2025).

**requirements.txt**
```text
pandas
numpy
xgboost
openpyxl
google-api-python-client
google-auth
google-cloud-storage
```

**Dockerfile**
```dockerfile
# Utilise une image Python de base
FROM python:3.11-slim

# Installe les dépendances système nécessaires pour XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     gcc     libgomp1 &&     rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Dépendances Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Script principal
COPY main.py /app/

# Commande par défaut
CMD ["python", "main.py"]
```

### 3.2 – Code de la Cloud Function (dossier : `carmila-trigger-function`)

**main.py** (fonction qui lance le job et charge vers BigQuery)  
> Collez ici le **script complet** de la fonction `run_prediction_and_load_to_bq` (appel Cloud Run Jobs + chargement du CSV du bucket vers BigQuery).

**requirements.txt**
```text
google-cloud-run
google-api-python-client
google-auth-oauthlib
google-cloud-bigquery
```

---

## Étape 4 : Déploiement de l'Infrastructure
> L'ordre des opérations est important.

### 4.1 – Créer le bucket Cloud Storage
Ce bucket stockera les fichiers CSV de sortie.
```bash
export BUCKET_NAME="carmila-results-$PROJECT_ID"
gcloud storage buckets create gs://$BUCKET_NAME   --location=$REGION_JOB   --project=$PROJECT_ID
```

*(Option recommandé – restreindre les droits au niveau du bucket)*
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME   --member="serviceAccount:$SA_JOB_EMAIL"   --role="roles/storage.objectAdmin"
```

### 4.2 – Créer l’ensemble de données et la table BigQuery
Créez d'abord un dataset, puis la table native (vide) qui recevra les données.
```bash
# Créer le Dataset
bq --project_id=$PROJECT_ID --location=EU mk --dataset $PROJECT_ID:carmila_data

# Créer la Table Native
bq --project_id=$PROJECT_ID mk --table $PROJECT_ID:carmila_data.predictions_carmila   nom_mall:STRING,cluster:STRING,date:DATE,donnees_reelles:FLOAT,donnees_predites:INTEGER
```

### 4.3 – Déployer le Job Cloud Run
Construisez, poussez et créez le job.
```bash
# Aller dans le dossier du job
cd carmila-prediction-job

export IMAGE_URI="$REGION_JOB-docker.pkg.dev/$PROJECT_ID/carmila-repo/carmila-xgb:latest"

# Créer le dépôt Artifact Registry (ignore l'erreur s'il existe déjà)
gcloud artifacts repositories create carmila-repo   --repository-format=docker   --location=$REGION_JOB   --project=$PROJECT_ID 2>/dev/null || true

# Authentifier Docker
gcloud auth configure-docker $REGION_JOB-docker.pkg.dev

# Build & Push
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# Créer le Job Cloud Run
export DRIVE_ID="<ID_DU_DOSSIER_DRIVE>"
gcloud run jobs create carmila-xgb   --region=$REGION_JOB   --image=$IMAGE_URI   --service-account=$SA_JOB_EMAIL   --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,DRIVE_FOLDER_ID=$DRIVE_ID"
```

### 4.4 – Déployer la Cloud Function
Déployez la fonction qui sert de pont entre le scheduler et le job.
```bash
# Aller dans le dossier de la fonction
cd ../carmila-trigger-function

export SA_SCHEDULER_EMAIL="scheduler-carmila@$PROJECT_ID.iam.gserviceaccount.com"

gcloud functions deploy trigger-carmila-job   --gen2   --region=$REGION_SCHEDULER   --runtime=python311   --source=.   --entry-point=run_prediction_and_load_to_bq   --trigger-http   --no-allow-unauthenticated   --service-account=$SA_JOB_EMAIL   --set-env-vars="GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION_JOB,DRIVE_FOLDER_ID=$DRIVE_ID,CLOUD_RUN_JOB_NAME=carmila-xgb,GCS_BUCKET_NAME=$BUCKET_NAME,BQ_DATASET=carmila_data,BQ_TABLE=predictions_carmila"

# Autoriser le scheduler à appeler cette fonction
gcloud functions add-invoker-policy-binding trigger-carmila-job   --region=$REGION_SCHEDULER   --member="serviceAccount:$SA_SCHEDULER_EMAIL"
```
> **Note** : Copiez l'**URL** de la fonction affichée à la fin du déploiement.

### 4.5 – Créer le Cloud Scheduler
Créez le déclencheur final.
```bash
export FUNCTION_URL="<URL_DE_LA_FONCTION_COPIÉE_CI-DESSUS>"

gcloud scheduler jobs create http trigger-carmila-job   --location=$REGION_SCHEDULER   --schedule="0 8 * * *"   --uri=$FUNCTION_URL   --http-method=POST   --oidc-service-account-email=$SA_SCHEDULER_EMAIL
```

---

## Étape 5 : Connexion à Looker Studio
1. Ouvrez Looker Studio (https://lookerstudio.google.com).
2. Créez une **nouvelle Source de Données** et choisissez le connecteur **BigQuery**.
3. Naviguez jusqu'à votre table : Projet **[VOTRE_PROJET]** → Dataset `carmila_data` → Table `predictions_carmila`.
4. Connectez cette source à un **nouveau rapport**.
5. Pour partager le rapport comme un modèle, modifiez son URL pour qu'elle se termine par `/copy`.

---

### Annexes – Conseils IAM (facultatif mais recommandé)
- Si l'exécution du **Cloud Run Job** échoue pour cause de permission, ajoutez au **SA du job** : `roles/run.developer` (ou conservez `roles/run.admin` comme ci-dessus).
- Si l'écriture dans GCS échoue, vérifiez les **bindings IAM au niveau du bucket** (étape 4.1).

---

**Fin du guide.**
