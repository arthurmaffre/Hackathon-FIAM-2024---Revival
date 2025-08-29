# ⚙️ Automatisation des Mises à Jour

Ce guide explique comment on peut configurer et automatiser les mises à jour du projet. Ces étapes me permettent d'assurer que tout reste à jour sans avoir à intervenir manuellement à chaque nouvelle version.

---

## 1️⃣ Clonage Initial

La première chose à faire est de cloner le dépôt sur la machine de déploiement. Comme **Git** est déjà installé, on peut simplement exécuter la commande suivante pour récupérer tout le projet :

```bash
git clone https://github.com/Thomas4390/McGillHackaton
```
## 2️⃣ Déplacement des Fichiers (Première Configuration)

Une fois le dépôt cloné, il faut maintenant déplacer certains fichiers pour permettre l'automatisation des mises à jour. Si c'est la première fois que je configure ce système, le fichier essentiel à déplacer est `on_start`.

Je vais donc utiliser la commande suivante pour déplacer ce fichier vers le bon répertoire :

```bash
cp /teamspace/studios/this_studio/McGillHackaton/start_file/on_start.sh /teamspace/studios/this_studio/.lightning_studio/on_start.sh

```
## ✅ Conclusion

En suivant ces étapes, on a configuré l'automatisation des mises à jour dans le projet. Le fichier `on_start.sh` a été correctement copié dans l'environnement de déploiement, assurant ainsi que les modifications sont prises en compte à chaque démarrage.

Désormais, le système est capable de gérer automatiquement les mises à jour et les cycles de déploiement sans nécessiter d'intervention manuelle supplémentaire. Cela garantit que le projet reste à jour et fonctionne de manière optimale.
