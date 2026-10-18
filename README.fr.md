[English](README.md) · Français

# Siliconed

*Siliconed — enhanced by silicone.*

**La génération d'images pour Apple silicon qui fait tout le calcul, le prouve, et ne prend jamais
plus de mémoire que son budget.**

Siliconed est une app macOS libre et gratuite pour la **génération d'images à partir de texte et
l'édition d'images par IA, en local**, sur les Mac Apple silicon (M1 et suivants). Elle fait tourner
des modèles de diffusion récents —
[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo),
[Anima Turbo](https://huggingface.co/circlestone-labs/Anima) et
[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) — en natif, en Swift et Metal : ni
Python, ni serveur, ni cloud, rien ne quitte votre Mac. Elle marche hors ligne une fois le modèle
installé.

![Licence : Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue)
![macOS 15+](https://img.shields.io/badge/macOS-15%2B-black)
![Apple silicon](https://img.shields.io/badge/Apple%20silicon-M1%20et%20suivants-black)

![Une photo d’été faite avec Z-Image Turbo — une Irlandaise rousse qui sourit sur une plage de l’ouest de l’Irlande, sa planche de surf sous le bras, en combinaison shorty — et son édition en bande dessinée, en cel shading, par Qwen-Image-2.1, sous un rideau qui glisse](docs/images/curtain.gif)

*Une photo de Z-Image Turbo, éditée par Qwen-Image-2.1 avec une phrase (« Convert this photo to a western
comic book style with cel shading »), comparées sous le rideau.*

## Pourquoi Siliconed

- **Vérifié contre les modèles d'origine.** Chaque étage (encodeur de texte, débruitage, VAE) est
  comparé, canal par canal, à l'implémentation de référence (diffusers / transformers en fp32 sur
  CPU). Écart à 1024² : 1,2·10⁻⁵ pour Z-Image. Ce qui ne se vérifie pas contre la référence n'est
  pas livré.
- **Tout en fp32, et rapide quand même.** Z-Image Turbo à 1024² en 113–119 s sur un M1 Pro,
  avec les 8 évaluations que demande sa recette ([mesuré à côté de Draw
  Things](#à-côté-de-draw-things), la référence sur Mac). Jusqu'à la 0.8.1 il n'en faisait que 7, et
  les temps publiés alors (82–85 s) valaient pour 7 ; corrigé dans la 0.8.2.
- **34 Go de modèle sur un Mac de 16 Go, dans un budget fixe.** Les poids sont lus depuis le disque
  au moment où ils servent. Siliconed ne prend jamais plus que les ¾ de la mémoire du Mac moins 2 Go
  (10 Go sur un Mac de 16 Go) : chaque rendu calcule ce qu'il lui faut avant de commencer, prend le
  plan le plus rapide qui y tient, et compare son propre pic à ce chiffre. Ce que les autres
  applications occupent leur appartient.
- **Fait pour être piloté par une IA.** `silicontrol` pilote l'app ouverte depuis un terminal, en
  sortie JSON ; `silicontrol help` est tout ce qu'un agent a besoin de lire.
- **Vos modèles, à votre façon.** Importez LoRA et fine-tunes depuis Civitai ou Hugging Face,
  installez en pleine taille, en 8 bits ou en 4 à 6 bits, et voyez chaque licence et la place disque
  avant le premier octet téléchargé.
- **Explorer avant de choisir.** Une grille d'esquisses rapides à travers graines, pas, forces de
  LoRA, modèles ou variantes du prompt ; la case choisie devient l'image finie.
- **Reproductible au bit près.** Même prompt, même graine, mêmes réglages, même version : la même
  image, octet pour octet. L'édition ci-dessus, refaite dans une autre session, est sortie
  identique. Une version peut changer les pixels : la 0.8.2 l'a fait pour Z-Image (sa huitième
  évaluation), ses images de la 0.8.1 et d'avant ne ressortent donc plus au bit.

## Les modèles

Seulement des modèles récents dont l'encodeur de texte est un modèle de langage — ni Stable
Diffusion, ni SDXL, ni Flux.1.

| Modèle | Éditeur | Ce qu'il fait | Licence |
|---|---|---|---|
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | texte vers image, image vers image, LoRA | Apache 2.0 |
| [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Qwen | texte vers image, édition par instruction avec 1 à 3 images, LoRA | Qwen Research (non commerciale) |
| [Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima) | CircleStone Labs × Comfy Org | texte vers image (illustration, peinture, anime ; pas de photoréalisme, par choix de l'éditeur), image vers image, LoRA | CircleStone Labs Non-Commercial ; ses images sont utilisables commercialement |
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea | texte vers image (des images esthétiques, de la photographie à la peinture), LoRA | Krea 2 Community : usage commercial sous 1 M$ de chiffre d'affaires annuel, un filtre de contenu obligatoire au déploiement |

Les versions Compact utilisent des poids 8 bits publiés par des tiers :
[unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF),
[Disty0/Z-Image-Turbo-SDNQ-int8](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-int8),
[unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8),
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) ; les versions Légères, le
GGUF Q4_K_M de [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) et
d'[unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) avec l'encodeur de
texte de la Compact. La Compact et la Légère d'Anima prennent le DiT en GGUF Q8_0 et Q4_K_M de
[vanes430/Anima-Turbo-V1.1-GGUF](https://huggingface.co/vanes430/Anima-Turbo-V1.1-GGUF) et gardent
l'encodeur de texte de l'éditeur en bf16 (aucune version 8 bits n'en est publiée).

Krea 2 Turbo ne télécharge rien de [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) :
ce dépôt demande un compte Hugging Face, et l'app n'a pas de connexion. Chacun de ses fichiers vient
d'une copie ouverte prouvée identique à celle de Krea : le DiT et l'encodeur de texte Qwen3-VL-4B en
bf16 de [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (le DiT, le même fichier ;
l'encodeur, égal tenseur pour tenseur), le VAE de [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
(le VAE fp32 de Krea est ce bf16 élargi, valeur pour valeur), le tokenizer de
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (le même fichier) ; les
fichiers de configuration de Krea sont embarqués tels que publiés. Sa Compact prend le DiT et
l'encodeur de texte en int8 convrot de [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) ; sa
Légère, le GGUF Q4_K_M de [vantagewithai/Krea-2-Turbo-GGUF](https://huggingface.co/vantagewithai/Krea-2-Turbo-GGUF)
avec l'encodeur de texte de la Compact. Chaque fichier est figé à une révision et vérifié par sha256
(`Sources/Siliconed/Forge/Installation.swift`).

![La fenêtre avec une image d’Anima Turbo : dans le rack, Anima Turbo et la LoRA inCase à 0,80 ; dans le canevas, une illustration peinte, une jeune Irlandaise rousse qui lit un livre sur une chaise en bois près d’une haute fenêtre, dans une vieille bibliothèque, une lumière d’automne à travers les carreaux, des rayonnages de livres à côté d’elle](docs/images/anima.jpg)

*Anima Turbo avec la LoRA inCase à 0,8 : « An oil painting of a young Irish woman with long red hair and
freckles, sitting by a tall window in an old library and reading a book. Warm afternoon light falls
across the wooden shelves; visible brushstrokes, muted earthy colors, painterly realism. » 1024², graine
44, 8 pas, 49 s sur un M1 Pro (Standard).*

<!-- KR7 : ici un rendu de Krea 2 Turbo (docs/images/krea2.jpg), fait dans l'app ; son texte
alternatif, puis sa légende : prompt, format, graine, pas, temps sur un M1 Pro (Standard). -->

## L'app

`Siliconed.app` génère des images à partir d'un prompt, les édite par instruction, et tient une file
et un historique. Elle suit la langue du système (anglais, français, allemand, espagnol ou italien).

### Créer

- **Quatre modèles.** Z-Image Turbo (Apache 2.0), Qwen-Image-2.1 Turbo (licence Qwen Research, non
  commerciale), Anima Turbo (licence CircleStone Labs, non commerciale ; ses images sont
  utilisables commercialement) et Krea 2 Turbo (licence Krea 2 Community : commerciale sous 1 M$ de
  chiffre d'affaires annuel, avec un filtre de contenu au déploiement). Tous quatre lisent le prompt
  avec un modèle de langage.
- **La chaîne, visible.** Le rack montre le pipeline de haut en bas (prompt, encodeurs, débruitage,
  décodeur), avec des fils typés et l'étage actif allumé ; chaque étage porte ses propres réglages.
- **Plus de détail** avec Z-Image et Qwen-Image-2.1 : « Détail : Normal · Plus · Maximum » ajoute
  de la texture fine, sans temps en plus.
- **Les variations** : des cousines d'une image, subtiles ou marquées, aux mêmes réglages.
- **Agrandir ×4** avec la super-résolution de macOS (macOS 26) : aucun prompt, rien d'ajouté, bords
  et textures affinés — 1024² devient 4096² en une ou deux secondes.
- **Les LoRA**, et **l'image vers image** avec Z-Image et Anima.

![La fenêtre : le rack à gauche (prompt, encodeur de texte, débruiteur, décodeur, chacun avec ses réglages, son temps et sa mémoire), une image Z-Image Turbo en 1248×832 à droite — une Irlandaise rousse sur une plage, sa planche de surf sous le bras —, l’historique en dessous](docs/images/window.jpg)

### Éditer

- **L'édition par instruction** avec Qwen-Image-2.1 : donnez-lui 1 à 3 images et une phrase
  (« remplace le ciel nuageux par un ciel bleu ») ; son encodeur voit les images. Dans l'app, une
  édition est plafonnée à une surface de 1024².
- **Le rideau** : faites glisser une ligne sur deux images, ou sur une édition ou un agrandissement
  et son original, pour voir ce qui a changé ; zoomez vers le curseur pour regarder un détail.
- **Un historique comme dans Photos** : flèches, ⌫ et ⌘Z, sélection multiple, comparaison côte à
  côte, export, glisser-déposer, Partager.

![Une édition par Qwen-Image-2.1 : la photo de la surfeuse en image 1 dans la boîte Images, l’instruction « Convert this photo to a western comic book style with cel shading », le résultat en cel shading dans le canevas](docs/images/edit.jpg)

![Une édition par Qwen-Image-2.1 à partir de deux images, sous le rideau qui glisse : à gauche la photo Z-Image Turbo de la surfeuse rousse en shorty sur la plage ; à droite la même femme, même plage, même lumière, avec le chemisier crème et la longue jupe verte de la liseuse peinte par Anima](docs/images/outfit.gif)

*Une édition qui puise dans une deuxième image : la surfeuse en image 1, la liseuse peinte par Anima en
image 2, et « Dress the woman of image 1 in the clothes of the woman in image 2 ». 1248×832, 297 s sur
un M1 Pro.*

### Explorer

Le panneau d'exploration construit une grille dans le canevas. Ses axes : les graines, les pas, la
force d'une LoRA, des LoRA à essayer, les modèles, les variantes écrites entre accolades dans le
prompt, chaque image d'une édition ou, en mode développeur, les solveurs. Chaque case est une esquisse, arrêtée après quelques pas, à
512 sur le petit côté (avec Qwen-Image-2.1, une image finie) ; dès qu'un modèle a rendu à cette
taille, le panneau dit ce que la grille coûtera sur ce Mac avant de la lancer. « Finir cette image »
rend la case choisie jusqu'au bout avec la même graine : l'image que son esquisse annonçait.

![Une exploration avec Anima Turbo et la LoRA inCase dans le canevas : quatre couleurs de cheveux (roux, blond, châtain, noir de jais) sur deux graines, une femme qui lit près d’une fenêtre de bibliothèque, des esquisses après 4 évaluations, les axes en étiquettes, « Finish This Image » sous la case choisie](docs/images/grid.jpg)

### La gestion des modèles

- **Rien n'est préinstallé.** Vous choisissez un modèle ; avant le premier octet téléchargé, l'app
  affiche sa licence et la place qu'il prend sur disque, et vous demande d'accepter. Chaque fichier
  est figé à une révision et vérifié par sha256.
- **Standard, Compact ou Légère.** Chaque modèle s'installe avec les poids de l'éditeur (Standard,
  le défaut), avec des poids 8 bits publiés par des tiers (Compact : 11,7 Go au lieu de 20,4 pour
  Z-Image, 17,3 Go au lieu de 31,2 pour Qwen-Image-2.1, 4,10 Go au lieu de 5,65 pour Anima, 19,5 Go
  au lieu de 34,4 pour Krea 2 ; une image légèrement différente, à peu près aussi rapide pour les
  trois premiers), ou avec le transformer en 4 à 6 bits tel qu'un tiers le publie (GGUF Q4_K_M ;
  Légère : 9,5 Go, 14,2 Go, 3,31 Go et 12,8 Go, le moins lu sur le disque à chaque pas ; un autre jeu
  de poids, une image qui peut se composer autrement). Pour Z-Image, Qwen-Image-2.1 et Krea 2,
  l'encodeur de texte de la Compact est lui aussi en 8 bits, et la Légère le reprend ; celui d'Anima
  reste le bf16 de l'éditeur dans les trois, faute de version 8 bits publiée. La Légère est
  présélectionnée sur un Mac de 8 Go de mémoire, la Standard ailleurs ; vous choisissez avant d'installer. Une version à la
  fois : passer à l'autre la remplace ; l'installée continue de rendre jusqu'à ce que la nouvelle
  soit complète, puis libère sa place — si le disque ne peut pas tenir les deux, l'app le dit avant
  l'acceptation.
- **Les imports Civitai et Hugging Face.** Les LoRA (formats Civitai, Comfy, kohya, diffusers) et le
  DiT d'un fine-tune (`.safetensors`, ou `.gguf` de Q4 à Q8_0), par glisser-déposer. Un fichier
  quantifié reste sur le disque tel que publié, jamais requantifié (Z-Image : 6,2 Go en 8 bits, environ
  5 Go en GGUF Q4_K_M, au lieu de 12,3 Go ; chaque pas un peu plus lent, le calcul reste en fp32) ;
  sous 4 bits, le fichier est refusé.
- **Statistiques et diagnostic.** La palette des statistiques (⌥⌘I) est un bouton de la barre d'outils ;
  « Signaler ma configuration » (ce que calcule la puce, mesuré, exporté en JSON — et les chemins que le
  moteur prend sur ce Mac) s'ouvre par « Chaîne » en tête du rack, orange tant que ce Mac n'est pas mesuré — le premier rendu le mesure, et de nouveau après une mise à jour.

Les modèles vivent dans `~/Library/Application Support/Siliconed` (modifiable avec `SILICONED_ROOT`).

![La fenêtre Modèles : Standard, Compact ou Légère, la licence et la place sur le disque montrées avant tout téléchargement](docs/images/models.jpg)

### Privé par construction

Le rendu se fait sur votre Mac. Le seul trafic réseau est le téléchargement du modèle que vous avez
accepté ; le diagnostic copie son rapport et ouvre un formulaire d'issue GitHub que vous choisissez de soumettre ou non.
Avant chaque rendu, le moteur vérifie le modèle, sa licence, le format et la mémoire qu'il lui faut,
et refuse d'emblée un rendu au-delà du budget de ce Mac.

## Télécharger

L'app est distribuée en `.dmg` sur la page
[Releases](https://github.com/gwenn-ha-dev/Siliconed/releases). Prérequis : macOS 15 ou plus, Apple
silicon. Les temps ci-dessous ont été mesurés avec 16 Go de mémoire.

## Les chiffres

Machine de référence : MacBook Pro M1 Pro (8 cœurs, 14 cœurs GPU), 16 Go. 1024², graine 42.

| | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo | Krea 2 Turbo |
|---|---|---|---|---|
| 1024², Standard | 113–119 s (8 pas ; 1024×1536 190 s) | 134 s (8 pas) ; 1024×1536 232 s ; une édition à 1 image 211 s, à 2 images 297 s | 43,7 s (8 pas) ; 512² 12,1 s ; 1024×1536 75,6 s | —<!-- KR : vitesse, Standard --> |
| 1024², Légère (4 à 6 bits) | 118–120 s | non remesuré depuis la 0.10 (tous les pas en pleine taille) | non mesuré | —<!-- KR : vitesse, Légère --> |
| pic mémoire, Standard / Légère | 2,6 Go / 2,0 Go | 1,8–2,7 Go / 1,5 Go | 1,4 Go (le moteur seul) / non mesuré | 3,07 Go (1024×1536 : 3,51 Go) / 1,35 Go, le moteur seul |
| écart à la référence fp32 (`model_out`) | 1,2·10⁻⁵ | 3,0·10⁻⁶ de l'exact fp64 (512²) | 1,4·10⁻⁵ de l'exact fp64 (1024²), la référence fp32 à 1,5·10⁻⁵ | —<!-- KR : écart --> |

Chaque temps est celui d'un rendu mis en file dans l'app par `silicontrol`, de la file au PNG :
Qwen-Image-2.1 mesuré le 15/10/2026 avec la 0.10, Z-Image le 14/10/2026 avec la 0.8.2. Le pic mémoire est le
`phys_footprint` de l'app, relevé toutes les 100 ms ; 0 page swappée sur la série. Les temps et le
pic d'Anima Turbo sont ceux du moteur seul, sans l'app, cache de rendu éteint (14–15/10/2026) ; les
pics de Krea 2 Turbo aussi (17/10/2026). Comment l'écart est mesuré : [La vérification](#la-vérification).

Jusqu'à la 0.8.1, Z-Image Turbo faisait 7 des 8 évaluations du DiT que demande sa recette (« 8 pas »
suivait un ancien calendrier de diffusers dont le dernier pas ne bougeait rien), les deux premières à
demi-taille à 1024². Corrigé dans la 0.8.2 : 8 évaluations en pleine taille. Les temps de Z-Image
publiés avant (82–85 s à 1024²) valaient pour le calendrier raccourci.

### À côté de Draw Things

[Draw Things](https://drawthings.ai) est la référence sur Mac ; ici, son moteur communautaire,
[draw-things-community](https://github.com/drawthingsai/draw-things-community) à `bc3970c`, avec
les réglages par défaut du modèle (UniPC trailing, CFG 1). La même machine, tout juste redémarrée, un
seul moteur chargé à la fois ; Z-Image Turbo, 1024², 8 pas (8 évaluations de chaque côté), graine 42,
trois rendus de chaque côté sur trois variantes du même prompt. Draw Things a été mesuré le 13/10/2026
(une séance de 70 minutes) et rien n'a changé de son côté depuis ; Siliconed le 14/10/2026, après
le correctif.

| | poids | temps | pic mémoire | lu sur le disque par rendu |
|---|---|---|---|---|
| Siliconed, Standard | les bf16 de l'éditeur, calculés en fp32 | 113–119 s | 2,6 Go | 51–58 Go |
| Siliconed, Légère | GGUF Q4_K_M | 118–120 s | 2,0 Go | 3,5–8,7 Go |
| Draw Things, Exact | f16 | 167–179 s | 5,3 Go | 99 Go |
| Draw Things, par défaut | 8 bits (`q8p`) | 142–164 s | 3,5–4,3 Go | 11 Go |
| Draw Things, 6 bits | 6 bits (`q6p`) | 136–138 s | 3,5 Go | 6–14 Go |

La comparaison porte sur le temps écoulé, pas sur une arithmétique identique : les solveurs et les
poids diffèrent, les images aussi. Le pic mémoire ne compte pas les poids projetés depuis le disque,
d'un côté comme de l'autre. Siliconed fait chauffer davantage la machine, ce qui découle du calcul
en fp32. Jusqu'à la 0.8.1, ce tableau opposait les 7 évaluations de Siliconed aux 8 de Draw Things ;
il compare maintenant 8 à 8.

**Dans le budget, mesuré.** Le pic `phys_footprint` de chaque modèle et ses pages swappées (delta
`vm_stat`) sont mesurés au pire cas du produit, sur un Mac de 16 Go où rien d'autre ne tourne : Qwen-Image-2.1 à 1024×1536 avec 3 références,
0 page swappée ; Z-Image à 1024×1536, pic 3,8 Go, 0 page swappée (son VAE décode en bandes, aux
mêmes bits).

Chaque côté doit être d'au moins 512 et multiple de 16, pour une surface d'au plus 1024×1536. Sous
512² les modèles sont hors de leur domaine.

## Face aux autres outils

- **[Draw Things](https://drawthings.ai)**, l'app native de référence sur Mac : son moteur est
  [mesuré plus haut](#à-côté-de-draw-things) sur la même machine. Siliconed calcule en fp32 et il est
  vérifié étage par étage contre l'implémentation de référence.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) ou [diffusers](https://github.com/huggingface/diffusers)
  en Python** : les mêmes modèles, sans environnement Python ni graphe de nœuds à câbler. Siliconed
  les implémente en natif et se vérifie contre diffusers et
  [transformers](https://github.com/huggingface/transformers) ; `silicontrol` couvre les scripts et
  l'automatisation.

## Le diagnostic communautaire

Une machine ne fait pas un banc. Le bouton **Signaler ma configuration** de l'app (*Report My
Configuration* en anglais) dresse la carte de ce que calcule la puce de votre Mac, en moins de 30
secondes et sans aucun modèle : les produits matriciels des blocs des modèles sur le GPU, sur l'unité
matricielle du CPU et sur les deux à la fois, des produits fp16 et bf16, l'attention par les trois
noyaux du moteur, l'élargissement des poids sur le CPU et sur le GPU — chaque résultat vérifié contre
le résultat exact calculé en double précision. Il copie ensuite le rapport JSON et ouvre un formulaire
d'issue GitHub où le coller (ou vous l'envoyez autrement) : puce, cœurs CPU et GPU, mémoire, macOS, drapeaux du CPU, chaque débit et s'il a
calculé juste, état thermique et alimentation avant et après. Rien n'est envoyé tant que vous ne
soumettez pas l'issue.

Les rapports remplissent ce tableau, une ligne par puce (TFLOP/s, en fp32 sauf mention) :

| Puce | Cœurs CPU | Cœurs GPU | Mémoire | GPU | Unité matricielle du CPU | Les deux | Attention | GPU fp16 | Source |
|---|---|---|---|---|---|---|---|---|---|
| M1 Pro | 8 | 14 | 16 Go | 3,5 | 2,2 | 5,1 | 2,5 | 3,4 | machine de référence |
| *votre ligne ici* | | | | | | | | | [rapport](https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml) |

Les temps de rendu de la machine de référence sont dans [Les chiffres](#les-chiffres).

## Pour les agents IA et les scripts : silicontrol

Pilote l'app ouverte depuis un terminal. Elle est livrée dans
`Siliconed.app/Contents/Helpers/silicontrol` ; Réglages → Ligne de commande la lie dans `/usr/local/bin`. Ce que
vous ajoutez apparaît dans la fenêtre de l'app, et il n'y a qu'une file, jamais deux rendus à la
fois.

```
silicontrol help                 la référence complète (toujours commencer par là)
silicontrol models
silicontrol add --wait --out out "a woman posing in a library"
silicontrol status
```

La sortie est en lignes JSON ; la dernière porte `"ok"`. Le dedans : `docs/REMOTE-CONTROL.fr.md`.

## Pour les développeurs

### Construire depuis les sources

macOS 15 ou plus, Apple silicon, Xcode 26 ou plus (l’app emploie les API de macOS 26 là où le système les a).

```
swift build                 la bibliothèque, l'app et silicontrol
swift test                  les tests unitaires purs, sans aucun poids
tools/app.sh                construit Siliconed.app à la racine du dépôt et l'ouvre
tools/app.sh --no-open      construit seulement
tools/dmg.sh                emballe l'app construite en .build/Siliconed-<version>.dmg
```

`swift test` ne demande rien d'autre : ni poids, ni Python.

Par défaut, l'app est signée ad hoc : elle tourne sur le Mac qui l'a construite. Pour la distribuer,
signez-la avec votre Developer ID :

```
tools/app.sh --sign "Developer ID Application: <nom> (<équipe>)"
```

Si `SILICONED_NOTARY_PROFILE` nomme un profil de trousseau `notarytool`, l'app est aussi notariée et
agrafée ; `tools/dmg.sh` signe et notarie alors l'image disque avec la même identité.

### La vérification

Chaque étage (encodeur de texte, débruitage, VAE) est comparé à l'implémentation de référence,
diffusers et transformers en fp32 sur CPU, sur la même entrée. L'écart se mesure canal par canal,
chaque canal rapporté à sa propre norme, pour qu'une faute confinée à un canal ne se cache pas dans
un chiffre global. Quand la référence fp32 porte sa propre erreur d'arrondi, l'une et l'autre sont
jugées contre un calcul fp64 du même étage, et le moteur ne doit pas s'en écarter de plus que la
référence plus 25 %. Le pic mémoire se lit sur `phys_footprint`, échantillonné pendant le rendu, le
swap sur `vm_stat`, et chaque chrono est encadré par une mesure témoin avant et après, car la machine
dérive de plusieurs pour cent d'un rendu à l'autre. Le banc d'oracles et le journal des mesures ne
font pas partie de ce dépôt ; ce qui est livré, ce sont les tests unitaires purs, et le diagnostic,
qui vérifie une machine contre des résultats exacts calculés sur son propre CPU.

## Documentation

- `docs/REMOTE-CONTROL.fr.md` : `silicontrol`, protocole et fonctionnement interne.
- `RELEASE-NOTES.fr.md` : ce qui change à chaque version.
- Les issues : [rapport de diagnostic, bug, idée](https://github.com/gwenn-ha-dev/Siliconed/issues/new/choose)
  (en anglais).

## Crédits

Les modèles appartiennent à leurs éditeurs (tableau ci-dessus), chacun sous sa propre licence. Les
implémentations de référence sont [diffusers](https://github.com/huggingface/diffusers) et
[transformers](https://github.com/huggingface/transformers), de Hugging Face ; le rééchantillonnage
des images reproduit le Lanczos de [Pillow](https://github.com/python-pillow/Pillow) au bit près. Les
LoRA et les fine-tunes viennent de la communauté, sur [Civitai](https://civitai.com) et
[Hugging Face](https://huggingface.co).

## Licence

Le code est sous [licence Apache 2.0](LICENSE). Les poids des modèles ne font pas partie de ce
dépôt : chacun est téléchargé par vous, sous sa propre licence, que l'app affiche avant de
l'installer.
