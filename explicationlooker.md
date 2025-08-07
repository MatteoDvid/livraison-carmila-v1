# Explications des graphiques — Dashboard Carmila

> Périmètre : filtres en haut (période, centres, clusters). Toutes les métriques se recalculent dynamiquement selon la sélection.

---

## 1) Vue « Executive » (KPI + 2 graphiques)

### 1.1 KPI (cartouches du haut)
- **Fréquentation prévue**  
  - **Fréquentation totale** : somme des prédictions sur la période sélectionnée (tous les centres sélectionnés).  
  - **Fréquentation journalière moyenne** : moyenne **par mall** et **par jour** →  
    \[ \text{FJM prédite} = \frac{\sum \text{Prédictions}}{\text{nombre de jours} \times \text{nombre de malls sélectionnés}} \]  
- **Fréquentation réelle** : mêmes définitions que ci‑dessus mais sur les valeurs observées (réelles).  
- **Écart**  
  - **Erreur moyenne** : grandeur moyenne des écarts quotidien *prédi − réel* (souvent mesurée en valeur absolue).  
  - **Taux de précision** : part de précision globale (ex. proche de **100 %** = prédictions proches de la réalité).  
  - **Écart‑type des erreurs journalières** : mesure la dispersion des erreurs jour par jour. **C’est un coefficient : plus il s’approche de *1*, plus c’est bon signe** (dispersion « normale » autour de l’erreur moyenne).

### 1.2 Prédictions **vs** Réalité (courbes)
Comparaison, jour par jour, des séries **prédi** et **réel**. Les croisements/écarts indiquent des jours où le modèle sous‑ ou sur‑estime la fréquentation.

### 1.3 Fréquentation par **cluster** (barres)
Répartition des totaux *réels* et *prédits* par cluster. Utile pour voir si un cluster est systématiquement sur‑ ou sous‑prédit.

---

## 2) Vue « Temporelle » (comparaison par années)
Graphique avec 4 courbes : **réel N**, **prédi N**, **réel N‑1**, **prédi N‑1**.  
- **Fréquentation N‑1** : total de l’année précédente sur la même période.  
- **Fréquentation N** : total observé sur la période sélectionnée.  
- **Variations** :  
  - *Variation (réel)* = \(\text{Fréquentation N} − \text{Fréquentation N‑1}\) rapportée à \(\text{Fréquentation N‑1}\).  
  - *Variation (prédi)* = \(\text{Prévision N} − \text{Fréquentation N‑1}\) rapportée à \(\text{Fréquentation N‑1}\).

---

## 3) Analyse par **centre commercial**
- **Tableau** : liste des centres avec *réel*, *prédi* et (souvent) un indicateur d’erreur (p. ex. MAE). Tri possible.  
- **Nombre de centres sélectionnés** : compteur dynamique en fonction des filtres.  
- **Fréquentation par centre** (barres) : totaux par centre pour repérer les plus contributifs et les centres à surveiller.

---

## 4) Performance modèle
### 4.1 « Courbe de Gauss » (histogramme des erreurs relatives)
Histogramme de **l’erreur relative de prédiction** (p. ex. \((\text{prédi}−\text{réel})/\text{réel}\)).  
- Une forme en « cloche » (gaussienne) centrée près de **0** indique un modèle **peu biaisé** (autant de sur‑ que de sous‑estimations).  
- **Queues épaisses** = nombreux cas extrêmes (jours atypiques, fermetures/ouvertures exceptionnelles, météo, événements…).

### 4.2 Erreurs de prédiction **au cours du temps** (série temporelle)
Série de l’erreur (souvent relative) jour par jour :  
- Une ligne proche de **0 %** = modèle globalement bien calibré.  
- Des **pics** = jours spécifiques (fermeture exceptionnelle, opération commerciale, jour férié, etc.).

---

## 5) Définitions clés (rappel rapide)
- **Fréquentation journalière moyenne (FJM)** : moyenne **par mall** et **par jour** sur la période et le périmètre sélectionnés.  
- **Écart‑type** : mesure de dispersion (plus il est grand, plus les valeurs varient autour de la moyenne).  
- **Écart‑type des erreurs journalières** : dispersion des erreurs quotidiennes ; **bon signe quand ≈ 1** (dispersion « normale »).  
- **Taux de précision** : indicateur global de proximité entre prédictions et réalité (plus il est élevé, mieux c’est).

> **Astuce lecture** : applique des filtres (clusters/malls/période) puis compare « Prédictions vs Réalité » et la « Courbe de Gauss ». Si la gaussienne se resserre autour de 0 et que la série d’erreurs reste plate, le modèle est bien calibré sur ce périmètre.
