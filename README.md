

# Data-Driven

## Présentation

Ce dépôt regroupe des travaux pratiques, projets et ressources liés au cours de **Deep Learning & Gen AI**. Il couvre la classification d'images, la segmentation, la détection d'objets, et l'utilisation de modèles avancés d'intelligence artificielle.

> **Ce projet est évolutif** : il s'enrichit au fil des cours et des expérimentations. Actuellement, le travail porte sur la détection d'objets avec YOLO (Atelier 4), mais d'autres modules seront ajoutés ou améliorés régulièrement.

## Organisation du projet

```text
├── Atelier 3/                # Segmentation d'images avec UNet
├── Atelier_4/                # Détection d'objets avec YOLOv8
├── IRM/                      # Classification de tumeurs cérébrales
├── resultats/                # Résultats et notebooks finaux
├── *.ipynb                   # Notebooks principaux
├── *.md                      # Documents de cours et explications
```

## Modules et Ateliers

- **Atelier 3 : UNet pour la segmentation**
	- Segmentation d'images médicales
	- Dataset : images et masques
- **Atelier 4 : YOLOv8 pour la détection d'objets**
	- Détection de véhicules et objets *(travail en cours)*
	- Dataset annoté (images/labels)
- **IRM Classification**
	- Classification de tumeurs cérébrales (gliome, méningiome, pituitaire, etc.)
	- Modèles CNN et analyse des performances
- **Autres Notebooks**
	- Expérimentations sur la classification de déchets, etc.

## Ressources

- Notebooks explicatifs et interactifs
- Datasets annotés pour chaque atelier
- Documents Markdown pour le support de cours


## Instructions d'utilisation

1. Cloner le dépôt :

```powershell
git clone https://github.com/mpigajesse/data-driven.git
```

2. Ouvrir les notebooks avec Jupyter ou VS Code
3. Installer les dépendances nécessaires (voir chaque notebook)
4. Suivre les instructions dans chaque atelier pour reproduire les résultats


## Environnements recommandés pour tester les notebooks

- **Python 3.8+** (idéalement 3.10 ou supérieur)
- **Plateformes compatibles :**
	- Jupyter Notebook
	- VS Code avec l'extension Jupyter
	- Google Colab
	- Kaggle Notebooks
	- JupyterLab
- **Environnements virtuels recommandés :**
	- `venv` (standard Python)
	- `conda` (Anaconda/Miniconda)
- **Dépendances principales :**
	- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
	- `torch`, `torchvision`, `tensorflow`, `keras`
	- `ultralytics` (pour YOLOv8)
	- Autres selon le notebook (voir instructions spécifiques)

## Technologies utilisées

- Python, Jupyter Notebook
- PyTorch, TensorFlow, Keras
- YOLOv8, UNet

## Auteur

Ce dépôt est maintenu dans le cadre du cours **Deep Learning & Gen AI**.

