# Atelier 2 — Deep Learning sur MNIST

Dépôt de travail pour l’atelier de Deep Learning (PyTorch). Les expériences sont prévues dans Kaggle ou Google Colab avec GPU. Les notebooks à compléter sont séparés par modèle; le notebook CNN fourni conserve les cellules exécutées et leurs résultats.

## Consignes de l’atelier

- **Partie 1 — CNN :** définir une architecture PyTorch et ses hyperparamètres, entraîner sur MNIST en mode GPU.
- **Partie 1 — Faster R-CNN :** réaliser l’expérience demandée et préciser l’adaptation nécessaire, car MNIST fournit des étiquettes de classe et non des boîtes de détection.
- **Comparaison :** comparer CNN et Faster R-CNN selon accuracy, F1, loss et temps d’entraînement.
- **Fine-tuning :** adapter les modèles pré-entraînés VGG16 et AlexNet à MNIST, puis comparer leurs résultats.
- **Partie 2 — ViT :** construire un Vision Transformer depuis zéro, l’entraîner sur MNIST, interpréter et comparer ses résultats.
- **Synthèse :** rédiger les apprentissages et la conclusion dans ce README.
- **Remise :** pousser le travail sur GitHub/GitLab et garder un bref rapport dans le README.

## Résultat déjà disponible — CNN

Résultats lus dans `notebooks/MNIST_CNN_Result.ipynb` (exécution Kaggle, GPU Tesla T4, jeu de test de 10 000 images) :

| Mesure | Résultat |
|---|---:|
| Loss test | 0,0125 |
| Accuracy test | 0,9956 (99,56 %) |
| F1 macro test | 0,9956 (99,56 %) |
| Durée d’entraînement | 339,7 s |
| Époques exécutées | 28 (early stopping) |
| Accuracy validation maximale affichée | 0,9967 (99,67 %) |

L’architecture CNN utilise trois couches convolutionnelles (32, 64 et 128 canaux), BatchNorm, ReLU, max pooling, dropout et deux couches linéaires. Le notebook sépare 54 000 images pour l’entraînement, 6 000 pour la validation et utilise les 10 000 images de test à la fin. Il inclut les courbes, le rapport par classe et des exemples de prédictions. Les détails par classe sont dans les sorties du notebook.

## État du travail

| Modèle / livrable | État | Fichier |
|---|---|---|
| CNN | Résultats disponibles | [`notebooks/MNIST_CNN_Result.ipynb`](notebooks/MNIST_CNN_Result.ipynb) |
| Faster R-CNN | À réaliser | [`notebooks/Faster_RCNN_MNIST.ipynb`](notebooks/Faster_RCNN_MNIST.ipynb) |
| VGG16 fine-tuning | À réaliser | [`notebooks/VGG16_FineTuning_MNIST.ipynb`](notebooks/VGG16_FineTuning_MNIST.ipynb) |
| AlexNet fine-tuning | À réaliser | [`notebooks/AlexNet_FineTuning_MNIST.ipynb`](notebooks/AlexNet_FineTuning_MNIST.ipynb) |
| ViT depuis zéro | À réaliser | [`notebooks/ViT_MNIST.ipynb`](notebooks/ViT_MNIST.ipynb) |
| Comparaison et synthèse finale | À compléter après les expériences | Cette section |

## Tableau comparatif à compléter

| Modèle | Accuracy test | F1 macro | Loss test | Temps (s) | GPU / remarques |
|---|---:|---:|---:|---:|---|
| CNN | 0,9956 | 0,9956 | 0,0125 | 339,7 | Tesla T4 |
| Faster R-CNN | — | — | — | — | À compléter; préciser la définition des boîtes |
| VGG16 fine-tuné | — | — | — | — | À compléter |
| AlexNet fine-tuné | — | — | — | — | À compléter |
| ViT | — | — | — | — | À compléter |

## Interprétation et synthèse personnelle

À compléter après les expériences :

1. Quel modèle obtient les meilleurs résultats et sur quelles métriques ?
2. Quel compromis observes-tu entre qualité, temps d’entraînement et complexité ?
3. Que t’a appris la comparaison entre CNN, modèles pré-entraînés, détecteur et ViT ?
4. Quelles difficultés as-tu rencontrées avec le GPU, les données ou l’adaptation des architectures ?

## Exécution

1. Ouvrir le notebook voulu dans Kaggle ou Google Colab.
2. Activer un accélérateur GPU.
3. Ajouter le jeu MNIST [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) si la plateforme le demande.
4. Exécuter les cellules dans l’ordre et noter les métriques dans le tableau ci-dessus.
5. Enregistrer les sorties pertinentes et compléter l’interprétation avant la remise.

Le notebook CNN fourni s’appuie sur le chemin d’entrée Kaggle `/kaggle/input`; il faudra adapter le chargement si tu l’exécutes dans Colab.
