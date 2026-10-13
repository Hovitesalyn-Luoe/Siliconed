[English](RELEASE-NOTES.md) · Français

# Siliconed — notes de version

## 0.8

### Rechercher les mises à jour

- **Dans le menu de l’app, à la demande seulement.** Siliconed demande à GitHub sa dernière version
  et la compare à celle qui tourne ; s’il en existe une plus récente, il ouvre sa page. Rien n’est
  vérifié en arrière-plan, rien n’est installé à votre place.

### Solveurs (mode développeur)

- **Euler ancestral, DPM++ 2M et UniPC**, en plus d’Euler, sur Z-Image Turbo et Qwen-Image-2.1 : les
  mêmes pas, les mêmes σ, une autre marche entre eux. Chacun est vérifié contre le scheduler de
  diffusers du même nom. Euler, par défaut, donne les mêmes images qu’avant, au bit près.
- **Une autre image, le même temps.** UniPC peut recomposer une image de Qwen-Image-2.1 ; le solveur
  ancestral peut aplatir un fond. Les éditions gardent l’image 1 sous les quatre. Aperçus et
  esquisses montrent l’estimation propre à chaque solveur. Dans le rack, la légende de l’historique,
  le PNG et `silicontrol add|grid --solver`.

### Les chiffres

- **Remesurés par l’app**, chaque rendu mis en file par `silicontrol` : Z-Image Turbo 1024² en 82 à
  85 s, Qwen-Image-2.1 en 118 s, une édition à une image en 212 s, à deux en 285 s ; 0 page
  swappée. Le README les pose à côté du moteur de Draw Things sur le même Mac.

Temps et mémoire mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.7.1

La 0.7 n’a jamais été publiée : la 0.7.1 la porte, avec le calcul de la mémoire rendu exact dans
les cas qu’il manquait.

### Mémoire : un budget fixe

- **Siliconed ne prend jamais plus que son budget** : les trois quarts de la mémoire du Mac moins
  2 Go (10 Go sur un Mac de 16 Go, 4 sur un Mac de 8). Il ne lit plus l’occupation du Mac : le même
  rendu sur le même Mac reçoit toujours la même réponse, quoi qu’il y ait d’autre d’ouvert. Ce que
  les autres applications occupent leur appartient.
- **Connu avant le premier pas.** Chaque rendu calcule la mémoire qu’il lui faut d’après son format,
  son prompt, ses images et ses LoRA, prend le plan le plus rapide qui tient dans le budget, et
  compare son propre pic à ce chiffre. Plus rien ne change de plan en cours de route. Ce chiffre est
  une borne haute : un long prompt sur Qwen-Image-2.1, un lot de plusieurs images et la mémoire de
  l’app elle-même sont comptés, et un rendu dont le besoin ne se calcule pas est refusé plutôt que
  lancé à l’aveugle.
- **Grisé d’avance.** Un format ou un modèle au-delà du budget de ce Mac est grisé avant le clic,
  avec le Mac qu’il demanderait : « demande un Mac de 24 Go ».
- **Les lots de Qwen-Image-2.1** ne gardent plus deux copies du modèle à la fois.
- Mêmes images, au bit près. Mesuré sur le M1 Pro : Z-Image Turbo en 1024² en 81 s,
  Qwen-Image-2.1 en 1024² en 116 s ; la plus grande édition (1024×1536, trois images) culmine à
  3,2 Go.
- Le rapport de diagnostic (schéma 4) donne le budget, le besoin de chaque étage et le pic mesuré en
  face.

### Sujets nommés (Expérimental)

- **Rien de gardé.** Les sujets vivent désormais en mémoire seulement, comme les images de la
  session : rien n’est écrit sur le disque, ils partent quand l’app se ferme. Un dossier de sujets
  écrit par la 0.6 est effacé au lancement.
- **Images ou Sujets.** La boîte au-dessus du prompt tient soit les images d’une édition (1 à 3),
  soit vos sujets (autant que vous voulez, le prompt en citant 3 au plus) — l’un ou l’autre, jamais
  les deux.
- **Le nom n’atteint jamais le modèle.** « @alice » devient « la personne de l’image 1 » : un nom lu
  comme du contenu donnait Alice au pays des merveilles.

Les temps et la mémoire sont mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.6

### Une seule boîte Images, au-dessus du prompt

- **Les images d’abord.** Sur Qwen-Image-2.1, la boîte qui tient les images à éditer s’appelle
  désormais Images et se place au-dessus du prompt : on pose l’image, puis on dit ce qu’on en fait.
  Ses fils passent à côté du prompt jusqu’aux encodeurs qui les lisent.
- **Un seul budget, compté une fois.** Avec les sujets nommés (Expérimental), les sujets vivent dans
  la même boîte, sous les images de l’édition : pour le modèle, un sujet cité est une image de plus.
  Le titre de la boîte compte les deux face aux 3 images du modèle, et passe au rouge au-delà.

### Mémoire

- **Un Mac au repos ne refuse plus Qwen-Image-2.1 en 1024².** Une génération simple se voyait
  compter la mémoire d’une édition à 3 images, et la marge de sécurité ignorait ce que macOS libère
  en compressant la mémoire des autres apps. Désormais un rendu compte le pic mesuré pour son propre
  nombre d’images, et la marge tient compte de cette compression : sur le Mac du signalement, de
  2,9 à 4,5 Go disponibles pour le même bureau. La règle du zéro swap ne change pas : tout reste
  mesuré au pire cas.

### Sujets nommés (Expérimental)

- **Comparer avec le sujet.** « @ana en western comic » garde le cadrage de la photo, comme une
  édition : le rideau sous l’image compare maintenant le résultat avec son image 1, qu’elle vienne de
  la boîte Images ou d’un sujet (son cadre, tel qu’il a été envoyé).

### Sous le capot

- Des noyaux et des réglages dépassés sont retirés (un noyau d’attention scalaire, un mode en demi-
  précision) : une seule version de chaque calcul vit, celle qui est vérifiée. Rien ne change dans
  les images.

Temps et mémoire mesurés sur un MacBook Pro M1 Pro de 16 Go.

## 0.5

### Qwen-Image-2.1 est désormais le Turbo officiel de Qwen

- **Un seul modèle, de son éditeur.** Qwen-Image-2.1 fait maintenant tourner le modèle distillé
  de Qwen lui-même, [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).
  Jusqu'à la 0.4, l'app faisait tourner le modèle de base sous la LoRA turbo d'un tiers (celle de
  Viggle) ; cette LoRA et le modèle de base disparaissent. Texte vers image, édition par
  instruction avec 1 à 3 images et vos propres LoRA marchent comme avant.
- **8 pas, sur le calendrier que Qwen publie.** Le nombre de pas est fixé à 8 : le mode à 9 pas
  disparaît, et tout autre nombre est refusé.
- **La Compact et la Légère suivent.** La Compact prend le Turbo 8 bits (int8 convrot) de
  Comfy-Org, la Légère le Turbo GGUF Q4_K_M d'unsloth ; l'encodeur de texte 8 bits de la Compact
  ne change pas. Sur le disque : 31,2 Go en Standard, 17,3 Go en Compact, 14,2 Go en Légère
  (33,2, 19,4 et 16,2 Go avant : la LoRA turbo n'est plus stockée).
- **Une installation existante est reconnue.** La fenêtre Modèles propose de « Finir
  l’installation… » : l'ancien transformer et les petits fichiers sont effacés, ceux du Turbo
  sont téléchargés dans la même version (Standard, Compact ou Légère). L'encodeur de texte est
  gardé : celui du Turbo est celui du modèle de base, au bit près, il ne se retélécharge donc pas.
  Les fine-tunes de Qwen-Image-2.1 importés pour le modèle de base ne peuvent plus rendre et sont
  effacés.
- **Même licence** : Qwen Research, non commerciale — le Turbo porte le texte du modèle de base.
- **Vérifié comme le reste.** Chaque étage est comparé à l'implémentation de référence de Qwen ;
  le moteur reste à 3,0·10⁻⁶ du calcul exact (fp64).

### Plus rapide que la 0.4 pour une image, plus lent pour une édition

Le Turbo prend 8 pas là où la LoRA tierce en prenait 6. Nous l'avons choisi quand même : c'est la
qualité que Qwen publie et vérifie, sans pièce tierce entre les deux qui pourrait changer ou
disparaître. Puis nous avons rendu ses pas moins chers, sans baisser la précision : tout est
toujours calculé en fp32 et vérifié contre le calcul exact.

- **Les deux premiers pas à demi-taille.** À partir de 1024², une génération calcule ses deux
  premiers pas en 512², puis agrandit, comme Z-Image le fait déjà. L'image change (la même graine
  donne une autre composition) mais reste propre. Pas sous une LoRA ni Détail, ni en édition.
- **L'attention partagée entre le GPU et le CPU.** Dans chaque couche, 13 des 32 têtes d'attention
  vont à l'unité matricielle du CPU pendant que le GPU calcule les autres.
- **De plus petits gains**, aux mêmes bits : le modèle est lu d'un seul flux, et des normalisations
  sont fusionnées.

Sur le Mac de référence :

- génération 1024² : ~120 s (~134 s en 0.4, ~160 s pour le Turbo sans ces gains) ;
- une édition avec 1 référence : ~210 s (~190 s), avec 2 références ~285 s ;
- le pire cas, 1024×1536 avec 3 références : ~515 s, 4,7 Go de mémoire au pic, sans aucun swap ;
- la Légère prend le même temps que la Standard.

### Installation

- **Plus rapide, et plus légère en mémoire.** La préparation d'un modèle téléchargé prend maintenant
  quelques secondes au lieu de plusieurs minutes, et le fichier suivant se télécharge pendant que le
  précédent se prépare. Sur une connexion rapide (85 Mo/s), Qwen-Image-2.1 s'installe en ~6 min 30 en
  Standard (~8 min en 0.4), ~3 min 30 en Compact (~5 min 30) et ~3 min en Légère (~4 min) : c'est
  désormais le réseau qui fixe la durée. Le modèle installé est le même, à l'octet près.
- **Pas de swap pendant l'installation.** Le pic de mémoire passe de 4,6 Go à 2,2–2,5 Go, et les
  fichiers fraîchement téléchargés ne poussent plus le reste du Mac hors de la mémoire : une
  installation qui pouvait swapper sur un Mac chargé ne le fait plus.

### Lots

- **Les lots ne swappent plus.** Chaque image d'un lot reconstruisait le modèle, et sur un Mac
  chargé un lot pouvait finir par swapper. Désormais toutes les images se calculent avec le même
  modèle, puis toutes se décodent : les mêmes images, sans swap, sur Z-Image et Qwen-Image-2.1.
- **Y compris avec votre LoRA.** Sur Qwen-Image-2.1, un lot avec une LoRA à vous garde le
  modèle, LoRA appliquée, d'une image à l'autre.

Temps mesurés sur un MacBook Pro M1 Pro, 16 Go.

## 0.4

### Installation

- **Standard, Compact ou Légère.** Z-Image Turbo et Qwen-Image-2.1 s'installent avec les poids de
  l'éditeur (Standard), avec des poids 8 bits publiés par des tiers pour le transformer et
  l'encodeur de texte (Compact : 11,7 Go au lieu de 20,4 pour Z-Image, 19,4 Go au lieu de 33,2 pour
  Qwen-Image-2.1), ou avec le transformer en 4 à 6 bits (GGUF Q4_K_M) et l'encodeur de texte de la
  Compact (Légère : 9,5 Go et 16,2 Go). La Légère est présélectionnée sur un Mac de 8 Go de mémoire.
  La version installée continue de rendre jusqu'à ce que la nouvelle soit complète.
- **Des téléchargements qui reprennent.** Les gros fichiers arrivent par morceaux, plusieurs à la
  fois ; un morceau qui cale est redemandé, et un téléchargement coupé reprend là où il s'était
  arrêté. Chaque fichier reste vérifié par sha256.
- **Signée et notariée.** L'app et l'image disque sont signées avec un Developer ID et notariées par
  Apple : elles s'ouvrent d'un double clic.

### Édition

- **Éditer depuis n'importe où.** Clic droit sur une image de l'historique, « Éditer avec
  Qwen-Image-2.1 », même quand le rack est réglé sur un modèle qui n'édite pas : le rack passe à
  l'éditeur et l'image devient l'image 1. Une vignette de l'historique glissée sur la boîte Éditer
  devient une référence.
- **Rideau original / éditée.** Une image éditée a un bouton qui la pose sur son original, avec une
  ligne à faire glisser pour voir exactement ce qui a changé. Échap le quitte.

### Images

- **Détail à 1024².** « Détail : Normal · Plus · Maximum » donne maintenant un résultat propre à
  1024² aussi, sur Z-Image et Qwen-Image-2.1 : sa force suit la taille de l'image, et il ne laisse
  plus de texture granuleuse en haute résolution.
- **Les variations suivent « Images ».** Les variations subtiles ou marquées d'une image en font
  autant que le champ « Images » du rack (1 par défaut), au lieu de toujours quatre.

### Exploration

- **Le modèle se choisit dans le panneau.** Le panneau d'exploration a son propre menu « Modèle ».
- **Une seule ligne d'état.** Le panneau dit où il en est en une ligne : « Esquisse à n pas sur N »,
  ou « Images finies, N pas » pour Qwen-Image-2.1, qui explore désormais en images finies (ses
  premières esquisses étaient floues).

### Modèles importés

- **Un 8 bits reste 8 bits.** Un fine-tune importé en 8 bits — fp8, int8, GGUF Q8_0 ou int8
  convrot — est gardé en 8 bits sur le disque, environ la moitié du fichier 16 bits (Z-Image :
  6,2 Go au lieu de 12,3 Go). Le calcul reste en fp32, et chaque poids est vérifié au bit près contre
  la conversion de l'éditeur du fichier. L'app importe les `.gguf` comme les `.safetensors`.
- **Le GGUF jusqu'à 4 bits.** Un fine-tune `.gguf` en Q4_0 à Q6_K est gardé tel que publié, ses blocs
  copiés entiers (Z-Image : environ 5 Go en Q4_K_M au lieu de 12,3 Go). Les fichiers sous 4 bits, et
  les fichiers 4 bits hors GGUF, sont refusés, avec la raison.

### Vitesse et mémoire

- **Z-Image est plus rapide à 1024²** : environ 82 s au lieu d'environ 107 s, et 153 s au lieu de
  168 s à 1024×1536, toujours sans aucun swap.
- **Un contrôle de mémoire sur l'état réel de la machine** : avant chaque rendu, l'app lit la mémoire
  qu'elle peut vraiment avoir à ce moment-là (pas seulement le total du Mac) et refuse, chiffres à
  l'appui, un rendu qui ne tiendrait pas — au lieu de le laisser swapper. Si la mémoire vient à
  manquer pendant un rendu, le moteur passe à un plan plus économe : la même image, un peu plus
  lentement, plutôt que de swapper.

### Langues

- **Allemand, espagnol et italien**, en plus de l'anglais et du français : l'app suit la langue du
  système.

Temps mesurés sur un MacBook Pro M1 Pro, 16 Go.
