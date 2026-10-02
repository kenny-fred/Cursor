# `Cursor`

## 1. L'objectif du projet

Le projet a pour objectif de contrôler le curseur d'une souris Windows uniquement avec les yeux à l'aide d'une webcam standard. 

Contrairement à de nombreux projets similaires, l'objectif n'est pas d'utiliser une caméra infrarouge spécialisée, un eye tracker Tobii onéreux, de simplement détecter si l'utilisateur regarde à gauche ou à droite, ou de reconstruire un regard 3D théorique.
---

## 2. La philosophie

La majorité des projets GitHub d'eye-tracking suivent une approche classique : une image webcam passe par un réseau neuronal pour estimer la direction du regard *(yaw/pitch)* et approximer la position écran.

Les limites de cette approche sont multiples : le regard 3D obtenu est souvent **approximatif**, la webcam ne connaît pas la profondeur réelle, deux personnes regardant le même point n'ont pas la même morphologie oculaire, et le moindre déplacement de la caméra change la donne.

### Notre approche :

Le système ne cherche pas à deviner le regard humain en général, mais apprend **votre** regard : Quand telle personne produit cette position d'iris et cette pose de tête, alors elle regarde à cet endroit précis.

---

## 3. Le calibrage et le format JSON

Au lieu de capturer des images massives pour alimenter un réseau neuronal lourd, `Cursor` construit son propre dataset personnel sous forme de mesures biométriques légères.

Lors du calibrage, un point s'affiche à l'écran, l'utilisateur le regarde, et le système effectue des captures pour enregistrer l'iris gauche, l'iris droit et la pose de tête.

### En gros :

Chaque cible produit un enregistrement structuré :

```json
{
  "screen": {
    "x": 640,
    "y": 360
  },
  "samples": [...]
}

```

Ce format présente des avantages majeurs : il est 

- extrêmement léger
- rapide, personnel, facile à nettoyer
- et garantit un **respect total de la vie privée** puisque aucune photo ni vidéo n'est conservée, seulement des nombres.

---

## 5. L'architecture v1

Le projet s'articule autour d'une suite de scripts aux rôles bien définis :

`camera_test.py` : Valide le matériel, ouvre la connexion avec la webcam, vérifie son ouverture, lit les images en boucle et gère une fermeture propre. Ce n'est pas un eye-tracker, mais une brique de validation hardware.



`eye_test.py` : Le premier véritable cerveau du système. Il s'appuie sur OpenCV, NumPy et MediaPipe pour récupérer la position des paupières, de l'iris et la forme des yeux sans entraîner d'IA lourde.



`calibration.py` : Le script central de calibration qui collecte les features biométriques et génère le dataset personnel sous format JSON.



`accuracy.py`: Évalue l'erreur et l'accuracy en comparant les données instantanées aux clusters de référence, mesurant le ratio Signal/Bruit (SNR).



`calib_test.py` : Assurent la conversion en temps réel du regard capturé par la webcam en mouvements de curseur Windows fluides.



---

## 6. Grid Map : 

### Anticipation et Stabilité

L'ajout de la grille de gestion dynamique (**les carrés bleus et jaune**s) introduit un mécanisme d'optimisation redoutable pour éviter les calculs superflus et stabiliser l'affichage :

`Les carrés jaunes` : Ils suivent de manière proactive les tendances de mouvement du curseur et l'intention de regard de l'utilisateur. Dès qu'une variation directionnelle est détectée dans les features de l'iris ou de la tête, le système** anticipe le déplacement pour éliminer toute sensation de latence perçue.**



`Les carrés bleus` : Dès que le regard se stabilise sur une zone cible, le système verrouille la position et réplique l'état stable sans relancer de calculs de régression lourds en boucle. Cela **élimine radicalement les redondances de calculs CPU, neutralise le jitter** (tremblement inhérent aux webcams) et **empêche le curseur de danser de manière erratique à l'écran.**

---

## 7. Comparaison v1

| Critères | Cursor | TrackyMouse | eViacam (Enable Viacam) | GazePointer |
| --- | --- | --- | --- | --- |
| **Qualité & Précision** | **Élevée (Signal/Bruit optimisé)** : Modèle hybride (iris + pose de tête) filtré, score de qualité visuel sur grille de calibration. | **Très bonne** : Hybride 2D/3D (suivi de points et influence du tilt) avec réajustement automatique. | **Bonne (Basée sur le mouvement facial)** : Suivi global de la tête, très stable mais demande un peu d'habitude. | **Moyenne / Délicate** : Tracking purement oculaire par webcam standard, sujet aux dérives et au bruit de capteur. |
| **Prix** | Gratuit (Projet personnel / FOSS) | Open Source (Gratuit) | Open Source (Gratuit / Donations) | Open Source / Gratuit |
| **Services & Fonctionnalités** | Contrôle fluide de la souris, rapport de calibration intelligent, architecture orientée performance. | Multi-modes de clics (yeux, clignement, bouche), raccourcis globaux, intégration web/CLI. | Émulation de souris universelle, assistant de configuration guidé, réglages de fluidité et d'accélération. | Suivi des yeux basique orienté accessibilité ou études comportementales légères. |
| **Structure & Architecture** | **Légère et optimisée** (~145 Mo `.exe`) : Extraction géométrique vectorielle, pas de Deep Learning lourd de pixels bruts, faible charge CPU. | **Moderne** : Application Electron (Node.js + Web technologies) avec surcouche d'affichage graphique (overlay). | **Native C++ / Légère** : Logiciel historique de bureau, extrêmement léger en ressources système. | **Moteur externe (GazeFlow)** : Repose sur un moteur de traitement d'image propriétaire/open source plus ancien. |
| **Design & Ergonomie** | Épuré, axé sur l'efficacité technique et la calibration visuelle par grille (24 points). | Interface moderne, intégration de barres d'outils et retours visuels en surimpression (*overlay*). | Interface utilisateur classique, utilitaire Windows ancré dans la tradition ergonomique des logiciels d'accessibilité. | Interface utilitaire minimaliste, axée sur les mires de réglage de la caméra. |
| **Avantages** | Absence de latence, rejet intelligent des *outliers*, robustesse face aux variations d'éclairage du décor, fluidité "limpide". | Polyvalence des modes de déclenchement (clin d'œil, bouche), grande communauté, support multiplateforme. | Fiabilité à toute épreuve, très faible consommation, idéal pour les personnes avec des mouvements de tête limités mais stables. | Utilise directement le regard (iris/pupilles) et non seulement la tête. |
| **Inconvénients** | Nécessite une phase de calibration rigoureuse sur grille au démarrage. | Peut souffrir de la complexité inhérente aux applications Electron (poids/mémoire vive). | Ne suit pas les yeux indépendamment de la tête (c'est la tête qui pilote le curseur, pas le regard pur). | Sensibilité extrême à la luminosité de la pièce et aux mouvements parasites de la tête (*jitter* oculaire important). |
| **Fonctionnement interne** | Extraction vectorielle des iris + angles de tête, puis Régression mathématique filtrée par seuil de bruit (SNR). | Suivi de facettes 2D combiné à l'inclinaison 3D de la tête + gestion d'états de clics par machine à états. | Algorithme de suivi de patron optique (*template matching*) centré sur les mouvements du visage/nez. | Analyse des reflets et de la position de la pupille par rapport aux coins de l'œil via webcam standard. |