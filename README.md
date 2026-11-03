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

## Résultats déjà disponibles

### CNN

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

### Faster R-CNN

Résultats lus dans `notebooks/Faster_RCNN_MNIST.ipynb` : entraînement distribué DDP sur deux GPU Tesla T4, modèle Faster R-CNN ResNet50-FPN, images 64x64, 59 000 images d'entraînement, 1 000 de validation et 10 000 de test. Après 8 époques (79,9 minutes), l'accuracy test est 99,22 % et le F1 macro 99,21 %. Le notebook ne rapporte pas de loss test; les losses train/validation sont affichées par époque. La définition des boîtes pour adapter MNIST à la détection est documentée dans le notebook.

### AlexNet fine-tuné

Résultats du notebook `notebooks/AlexNet_FineTuning_MNIST.ipynb` : accuracy test **99,25 %**, F1 macro **99,24 %**, loss test **0,0236** et durée **3,8 min** (10 époques, entraînement distribué sur deux Tesla T4).

### VGG16 fine-tuné

Résultats du notebook `notebooks/VGG16_FineTuning_MNIST.ipynb` : accuracy test **99,49 %**, F1 macro **99,49 %**, loss test **0,0226** et durée **6,0 min** (early stopping après 9 époques, entraînement distribué sur deux Tesla T4).

### Vision Transformer (ViT)

Résultats du notebook `notebooks/ViT_MNIST.ipynb` : accuracy test **87,92 %**, loss test **0,3746** et durée **62,6 s** après 5 époques, sur deux GPU Tesla T4 avec DataParallel. Le notebook ne rapporte pas le F1 macro.

## État du travail

| Modèle / livrable | État | Fichier |
|---|---|---|
| CNN | Résultats disponibles | [`notebooks/MNIST_CNN_Result.ipynb`](notebooks/MNIST_CNN_Result.ipynb) |
| Faster R-CNN | Résultats disponibles | [`notebooks/Faster_RCNN_MNIST.ipynb`](notebooks/Faster_RCNN_MNIST.ipynb) |
| VGG16 fine-tuné | Résultats disponibles | [`notebooks/VGG16_FineTuning_MNIST.ipynb`](notebooks/VGG16_FineTuning_MNIST.ipynb) |
| AlexNet fine-tuné | Résultats disponibles | [`notebooks/AlexNet_FineTuning_MNIST.ipynb`](notebooks/AlexNet_FineTuning_MNIST.ipynb) |
| ViT depuis zéro | Résultats disponibles | [`notebooks/ViT_MNIST.ipynb`](notebooks/ViT_MNIST.ipynb) |
| Comparaison et synthèse finale | À compléter après les expériences | Cette section |

## Tableau comparatif

| Modèle | Accuracy test | F1 macro | Loss test | Temps (s) | GPU / remarques |
|---|---:|---:|---:|---:|---|
| CNN | 0,9956 | 0,9956 | 0,0125 | 339,7 | Tesla T4 |
| Faster R-CNN | 0,9922 | 0,9921 | Non rapportée | 4 794 s (79,9 min) | 2 Tesla T4, 8 époques; adaptation MNIST détection |
| VGG16 fine-tuné | 0,9949 | 0,9949 | 0,0226 | 360 | 2 Tesla T4; early stopping à 9 époques |
| AlexNet fine-tuné | 0,9925 | 0,9924 | 0,0236 | 228 | 2 Tesla T4; 10 époques |
| ViT | 0,8792 | Non rapporté | 0,3746 | 62,6 | 2 Tesla T4; DataParallel; 5 époques |

## Interprétation et synthèse personnelle

Sur les exécutions disponibles, le CNN obtient la meilleure accuracy (99,56 %) et la meilleure loss rapportée (0,0125). VGG16 suit avec 99,49 % d’accuracy et 99,49 % de F1 macro. AlexNet atteint 99,25 % d’accuracy, et Faster R-CNN 99,22 %. Le ViT atteint 87,92 % après cinq époques : ce résultat est nettement inférieur dans cette configuration et ne permet pas de conclure que les ViT sont généralement moins performants; il suggère plutôt que cette implémentation ou son entraînement demande davantage d’ajustements ou d’époques.

Les durées rapportées ne sont pas directement comparables : CNN a été entraîné sur une Tesla T4 pendant 339,7 s, tandis que Faster R-CNN, AlexNet et VGG16 ont utilisé deux Tesla T4 avec des configurations et des procédures différentes. Le F1 du ViT n’est pas fourni, et Faster R-CNN ne rapporte pas de loss test.

**Limite de protocole à prendre en compte :** le notebook ViT affiche la loss du jeu de test à chaque époque, alors que le test devrait idéalement rester réservé à l’évaluation finale. Il faut donc éviter de présenter ce résultat comme une comparaison parfaitement contrôlée; pour une comparaison plus rigoureuse, choisir les réglages à partir d’un jeu de validation distinct, puis évaluer le test une seule fois.

### Ce que j’ai appris

J’ai appris qu’un CNN est bien adapté à la classification d’images comme MNIST : il apprend progressivement les motifs visuels utiles et a obtenu ici le meilleur résultat, avec 99,56 % d’accuracy. Le fine-tuning de modèles pré-entraînés peut aussi donner d’excellents résultats : VGG16 atteint 99,49 % et AlexNet 99,25 %. Faster R-CNN est conçu pour localiser des objets en plus de les classer; son utilisation sur MNIST demande donc de définir des boîtes autour des chiffres. Enfin, un ViT traite l’image en patches et utilise l’attention; dans cette expérience, son résultat après cinq époques reste plus faible, ce qui montre que l’architecture et les réglages d’entraînement comptent beaucoup.

### Compromis observé

Sur ces essais, VGG16 est proche du CNN en accuracy, mais a pris 6 minutes contre 339,7 secondes pour le CNN; AlexNet a pris 3,8 minutes et Faster R-CNN environ 79,9 minutes. Ces temps ne constituent pas une comparaison parfaitement équitable, car les expériences n’utilisent pas toutes le même nombre de GPU ni les mêmes configurations. Les modèles plus complexes ou adaptés à la détection peuvent coûter davantage en calcul sans améliorer la classification de chiffres simples.

### Difficultés et améliorations proposées

L’adaptation de Faster R-CNN est délicate parce que les données MNIST ne fournissent pas de boîtes de détection. Le ViT n’a été entraîné que cinq époques et son notebook suit la loss du test à chaque époque; pour une comparaison plus rigoureuse, je séparerais un jeu de validation pour suivre l’entraînement, réserverais le test à une seule évaluation finale, puis ajusterais les hyperparamètres et prolongerais l’entraînement du ViT. Je rapporterais aussi les mêmes métriques pour chaque modèle et utiliserais une configuration GPU comparable.

## Exécution

1. Ouvrir le notebook voulu dans Kaggle ou Google Colab.
2. Activer un accélérateur GPU.
3. Ajouter le jeu MNIST [`hojjatk/mnist-dataset`](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) si la plateforme le demande.
4. Exécuter les cellules dans l’ordre et noter les métriques dans le tableau ci-dessus.
5. Enregistrer les sorties pertinentes et compléter l’interprétation avant la remise.

Le notebook CNN fourni s’appuie sur le chemin d’entrée Kaggle `/kaggle/input`; il faudra adapter le chargement si tu l’exécutes dans Colab.
