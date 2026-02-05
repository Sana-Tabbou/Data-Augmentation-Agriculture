<img width="300" height="150" alt="image" src="https://github.com/user-attachments/assets/e42542e6-afc8-4b0b-b619-ea1741e565a9" />
<img width="760" height="150" alt="image" src="https://github.com/user-attachments/assets/7c1b8693-2d86-4a14-8431-3883ed04abef" />




# Framework d'Augmentation de Données pour l'Imagerie Agricole

**Projet de Fin d'Études **  
*Développement de méthodes d'augmentation de données pour l'imagerie agricole*

---

## Présentation du projet

Ce projet développe un **framework d'augmentation de données à trois niveaux** spécialement conçu pour les applications d'IA agricole. Il répond au manque de données étiquetées dans le domaine agricole en générant des images d'entraînement réalistes et variées grâce à des transformations géométriques, photométriques et spécifiques au domaine.

## Objectifs

- **Étendre artificiellement les jeux de données** agricoles pour améliorer l'entraînement des modèles d'IA
- **Construire un pipeline Python modulaire** pour une réutilisation et une adaptation faciles
- **Générer des variations réalistes** qui préservent les caractéristiques agricoles essentielles
- **Améliorer la robustesse des modèles** face aux conditions réelles (lumière, météo, angles)

## Outils & Technologies

- **Langages** : Python
- **Bibliothèques** : OpenCV, TensorFlow/Keras, NumPy, Matplotlib, PIL
- **Environnements** : Jupyter Notebook, Google Colab, Git
- **Méthodologie** : Développement itératif, phases de validation, gestion des risques

## Structure du Framework

Notre stratégie d'augmentation est organisée en trois niveaux :

### 1. Méthodes Simples
   - Rotation, retournement, redimensionnement
   - Ajustement de luminosité et de contraste
   - Recadrage

### 2. Méthodes Avancées
   - Flou de mouvement, ajout de bruit
   - Amélioration locale du contraste (CLAHE)
   - Réglage avancé des couleurs et de la saturation

### 3. Méthodes Agricoles Spécifiques
   - Simulation météo (brouillard, pluie, ombres de nuages)
   - Effets de conditions du sol (humide, sec)
   - Variations saisonnières et d'éclairage (lumière matinale, effet sécheresse)

## Auteurs :
  - Sana TABBOU

## Tuteur encadrant :
  - M. Frédéric COINTAULT

