# Atelier 3 — Segmentation d’Images avec U-Net

*Ce document est la conversion du PDF « Atelier 3_ Segmentation d’Images avec U-Net.pdf » en format Markdown pour faciliter la lecture, l’édition et l’intégration dans votre environnement virtuel.*

---

## Table des matières
- Introduction
- Objectifs
- Prérequis
- Accès aux données
- Prétraitement
- Modélisation U-Net
- Entraînement
- Évaluation
- Visualisation
- Sauvegarde

---

> **Remarque :** Ce document est une base. Pour une conversion complète, veuillez préciser si vous souhaitez inclure toutes les sections, images, schémas, ou seulement le texte principal. Les titres et la structure sont adaptés pour un usage pédagogique et reproductible dans un notebook ou un README.

---

## Introduction
La segmentation d’images médicales est une tâche clé en deep learning. L’architecture U-Net est particulièrement adaptée à ce type de problème.

## Objectifs
- Comprendre le pipeline de segmentation avec U-Net
- Préparer les données (images/masques)
- Implémenter et entraîner un modèle U-Net
- Évaluer et visualiser les résultats

## Prérequis
- Python ≥ 3.8
- TensorFlow ≥ 2.x
- Bibliothèques : numpy, pandas, matplotlib, scikit-image, opencv, albumentations
- Accès à Google Colab ou un environnement local compatible GPU

## Accès aux données
- Les images et masques sont stockés dans Google Drive, dossier `datasets`.
- Structure attendue :
  - `datasets/images/train`, `datasets/images/val`, `datasets/images/test`
  - `datasets/masks/train`, `datasets/masks/val`, `datasets/masks/test`

## Prétraitement
- Redimensionnement des images et masques
- Normalisation
- Augmentation (optionnelle)

## Modélisation U-Net
- Construction du modèle avec Keras
- Fonction de perte : Dice + Binary Crossentropy
- Métriques : Accuracy, MeanIoU, Dice

## Entraînement
- EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Sauvegarde du meilleur modèle dans Drive

## Évaluation
- Calcul des métriques sur le jeu de test
- Visualisation des prédictions

## Visualisation
- Affichage des images, masques et prédictions côte à côte

## Sauvegarde
- Modèle final et historique d’entraînement sauvegardés dans Drive

---

> **Pour une conversion complète avec le contenu détaillé, merci de préciser si vous souhaitez inclure les exemples de code, schémas, ou uniquement le texte explicatif.**
