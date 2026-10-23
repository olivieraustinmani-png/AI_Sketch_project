# AI Sketch — Intelligent Drawing and Object Recognition Tool

Application de bureau en français pour dessiner une forme et prédire `house`, `tree`, `car`, `cat`, `sun`, ou `unknown` (Inconnu).

## Fonctionnalités

- Canevas souris, épaisseur du pinceau réglable, effacement et sauvegarde PNG.
- Prédiction à partir du modèle entraîné ; classe et score indicatif affichés.
- Détection d'un canevas vide et messages si modèle ou dépendances manquent.
- Prétraitement partagé : niveaux de gris, inversion en traits clairs sur fond sombre, cadrage, centrage, taille 64×64 et pixels normalisés entre 0 et 1.

La reconnaissance est déterministe pour un même modèle et un même dessin. Elle n'est pas exacte à tous les coups. Les dessins hors domaine sont rejetés à partir d'exemples « unknown » et d'un seuil ajusté sur validation, mais cette protection ne couvre pas tous les objets imaginables. Le score affiché n'est pas une probabilité calibrée.

## Prérequis et installation

Environnement constaté : Windows 11, Python 3.14.4 64 bits. `requirements.txt` installe seulement les dépendances de l'application. La prise en charge TensorFlow dépend de la version de Python ; utilisez Python 3.11 ou 3.12 pour l'entraînement avec le fichier distinct `requirements-training.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

L'interface utilise PyQt5, NumPy et Pillow. Un modèle NumPy entraîné est actuellement inclus dans le dossier du projet, donc la reconnaissance fonctionne aussi sous Python 3.14 sans TensorFlow.

Pour entraîner le CNN Keras, installez Python 3.12 puis créez un environnement séparé :

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

## Lancer

```powershell
python app.py
```

Dans l'application, dessinez, réglez l'épaisseur puis utilisez **Effacer**, **Enregistrer PNG** ou **Reconnaître**. Les images enregistrées par défaut sont proposées dans `saved_drawings/`.

## Données et entraînement

Le script utilise les fichiers NDJSON simplifiés officiels Google [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), téléchargés depuis leur stockage public. Les cinq classes reconnues sont house, tree, car, cat et sun. La sixième étiquette `unknown` est entraînée avec des dessins de apple, banana, fish, airplane, flower et cup. Les six étiquettes de sortie sont définies dans `labels.json`; les fichiers sélectionnés sont mis en cache dans `data/quickdraw/`.

```powershell
python train_model.py --per-class 1000
```

La valeur indique le maximum d'exemples par classe connue. Le partage stratifié 70/15/15 sépare entraînement, validation et test. Le script rasterise les traits et applique le prétraitement commun. Le moteur NumPy compare les dessins avec les exemples d'entraînement et prédit « unknown » lorsque les exemples hors domaine sont plus proches ou que le dessin est trop éloigné des classes connues. Pour entraîner sans dépendance additionnelle :

```powershell
python train_model.py --per-class 100 --unknown-per-category 30
```

Le modèle NumPy est sauvegardé dans `ai_sketch_model.npz`; les effectifs, précision, rappel « unknown » et seuil sont écrits dans `training_metrics.json`. `--unknown-per-category` contrôle le nombre d'exemples négatifs téléchargés pour apple, banana, fish, airplane, flower et cup. Une connexion Internet est requise pour compléter le cache. Les données brutes ne sont pas incluses dans le dépôt.

## Tests et état des livrables

Vérifié sous Python 3.14.4 : l'application démarre et le canevas a été testé. Dernier entraînement réel : 600 dessins, répartis en 420 entraînement, 90 validation et 90 test. Précision test mesurée : 67,8 %. Le rappel « unknown » est de 46,7 % (7/15 exemples inconnus). Les performances par classe restent inégales; ces résultats sont ceux d'un prototype et ne garantissent pas la reconnaissance d'un dessin utilisateur. Aucun résultat n'est inventé.

## Limites et difficultés

Le modèle traite cinq classes connues et une classe « unknown ». Les exemples négatifs d'entraînement proviennent de pomme, banane, poisson, avion, fleur et tasse; des objets hors de ces catégories peuvent encore être mal classés. L'application ne peut pas garantir une reconnaissance exacte : elle dépend de la diversité et de la taille des données. Le rejet utilise les exemples proches et un seuil réglé sur validation. Ajouter des dessins étiquetés à ces six catégories hors domaine améliore la couverture; le score affiché reste indicatif.

## Rapport

Le rapport universitaire est dans `report/project_report.docx`. Les noms des membres du groupe et leurs contributions restent à compléter.
