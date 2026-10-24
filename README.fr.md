[English](README.md) · Français

# Siliconed

*Siliconed — enhanced by silicone.*

**La génération d'images pour Apple silicon qui mesure votre Mac, choisit ses noyaux pour cette
puce, fait tout le calcul, le prouve, et ne prend jamais plus de mémoire que son budget.**

Siliconed est une app macOS libre et gratuite pour la **génération d'images à partir de texte et
l'édition d'images par IA, en local**, sur les Mac Apple silicon (M1 et suivants). Elle fait tourner
des modèles de diffusion récents —
[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo),
[Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
[Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) et
[Anima Turbo](https://huggingface.co/circlestone-labs/Anima) — en natif, en Swift et Metal : ni
Python, ni serveur, ni cloud, rien ne quitte votre Mac. Elle marche hors ligne une fois le modèle
installé.

![Licence : Apache 2.0](https://img.shields.io/badge/licence-Apache%202.0-blue)
![macOS 15+](https://img.shields.io/badge/macOS-15%2B-black)
![Apple silicon](https://img.shields.io/badge/Apple%20silicon-M1%20et%20suivants-black)

![Une photo ensoleillée de Krea 2 Turbo — une Irlandaise rousse en combinaison shorty qui sort des vagues en riant, sa planche de surf sous le bras, à contre-jour d’un soleil bas — et son édition en bande dessinée western, en cel shading, par Qwen-Image-2.1, sous un rideau qui glisse](docs/images/curtain.gif)

*Une photo de Krea 2 Turbo, éditée par Qwen-Image-2.1 avec une phrase (« Convert this photo to a western
comic book style with cel shading », graine 42, 215 s sur un M1 Pro), comparées sous le rideau.*

## Krea 2 Turbo, le plus gros des quatre

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), de Krea, est le quatrième modèle, arrivé
dans la 0.12 : un DiT de 12,9 milliards de paramètres, le plus gros des quatre, pour des images
esthétiques, de la photographie à la peinture. Il lit le prompt avec Qwen3-VL-4B et fait toute sa
recette de 8 évaluations en fp32 sur les poids bf16 de l'éditeur : **1024² en 145,6 s, 1024×1536 en
216,9 s sur un M1 Pro de 16 Go** avec le [rendu plus rapide](#rendu-plus-rapide), allumé par défaut
(221,1 s et 355,3 s avec toutes les évaluations en pleine taille), ses 34,4 Go lus depuis le disque
dans le budget mémoire. La sortie de son débruiteur est à 2,6·10⁻⁴ de la référence fp32 à 1024².
Il s'installe en Standard, Compact ou Légère
(34,4, 19,5 ou 12,8 Go), depuis des copies ouvertes prouvées identiques aux fichiers de Krea
([pourquoi](#les-modèles)), sous la licence Krea 2 Community : usage commercial sous 1 M$ de chiffre
d'affaires annuel, un filtre de contenu obligatoire au déploiement.

![Une photo de Krea 2 Turbo en 1248×832, en contre-plongée : une jeune Irlandaise aux longs cheveux auburn ondulés et aux taches de rousseur, en combinaison shorty à manches courtes, qui sort des vagues en riant, sa planche de surf sous le bras, à contre-jour d’un soleil doré bas avec un reflet d’objectif, des embruns lumineux, une eau turquoise et un ciel bleu profond](docs/images/krea2.jpg)

*Krea 2 Turbo, Standard, 1248×832, 8 pas, graine 3001, 229 s sur un M1 Pro de 16 Go : « Editorial
surf photograph, low angle, of a young Irish woman with long wavy auburn red hair, pale freckled skin
and green eyes, in a short-sleeved shorty wetsuit, running out of the surf laughing, surfboard under
her arm, on a rare sunny day in the west of Ireland. Golden late-afternoon backlight, sun flare,
spray glowing in the light, turquoise water, deep blue sky, high contrast, vivid colors, sharp
focus, shot on a 50 mm lens. »*

## Un moteur mesuré sur votre Mac

**Deux Mac ne calculent pas pareil, alors Siliconed ne devine pas.** Avant le premier rendu qui suit
chaque installation ou mise à jour — et dès que le Mac ou macOS change —, il mesure le Mac sur lequel
il tourne, en une minute environ et sans aucun modèle. Il chronomètre les vrais produits matriciels
des blocs des DiT sur le GPU, sur l'unité matricielle du CPU (AMX), sur le Neural Engine et sur le
GPU à côté de chacun, l'attention par trois noyaux, des produits lus directement sur les poids int8
et Q8_0 publiés, et des produits bf16 sur les unités tensorielles du GPU, et vérifie chaque sortie
contre le résultat exact. Ce rapport devient **la carte de ce Mac**, et la carte choisit les
noyaux : un noyau ne s'allume que s'il a calculé juste et bat le défaut d'au moins 5 % (davantage
si la machine a dérivé pendant la mesure).

Ce que la carte a décidé sur le M1 Pro des chiffres plus bas, 16 Go, le 23/10/2026 (app 0.14) :

| Noyau | Décision | Contre le défaut |
|---|---|---|
| co-exécution GPU + unité matricielle du CPU (AMX) | **allumée** | ×1,26 |
| le Neural Engine à côté du GPU | **allumé** | ×1,42 |
| attention partagée par têtes entre le GPU et le CPU | **allumée** | ×1,48 |
| attention flash | éteinte, dans la marge | ×1,03 |
| produits sur les poids Q8_0 publiés | éteints | ×0,37 |
| produits bf16 sur les unités tensorielles du GPU | éteints | ×0,47 |

Les produits sur les poids int8 publiés (la Compact de Krea 2 et de Qwen-Image-2.1) ont gagné sur
cette puce à la mesure de la 0.12 (18/10/2026) et se sont allumés. L'AMX et le Neural Engine se
partagent les produits forme par forme : le Neural Engine prend une forme partout où il garde 90 %
de la vitesse de l'AMX, pour l'énergie qu'il épargne. C'est un seul Mac : une autre puce mesure les
siens et en décide autrement, et son rapport peut remplir une ligne du tableau du
[diagnostic communautaire](#le-diagnostic-communautaire). La carte achète de la vitesse, jamais une
autre image au-delà de l'arithmétique qu'elle a vérifiée ; le fp16 du Neural Engine, c'est vous qui
le choisissez, par modèle (ci-dessous). Aide → **Mesurer ce Mac** montre la carte et permet de la
partager.

### Le Neural Engine

Le Neural Engine du Mac calcule en fp16. Là où la carte de ce Mac l'a allumé, il prend une partie des produits matriciels du DiT à côté du GPU, forme par forme : chaque colonne de poids et chaque ligne d'activations est remontée d'une puissance de deux dans la plage du fp16 puis ramenée en fp32, et les lignes qui débordent retournent au GPU. Les encodeurs de texte et le VAE ne s'en servent jamais. Le tag **fp32** du module DiT du rack le bascule, par modèle : **éteint par défaut sur la Standard**, qui reste les poids de l'éditeur calculés en fp32, l'image vérifiée contre la référence ; **allumé par défaut sur la Compact et la Légère**, dont les poids en 8 et 4 bits font déjà une autre image que celle de la Standard. Le tag affiche **fp32 · ANE** quand il est allumé, et une image calculée avec lui porte `neural-engine` dans ses métadonnées. Là où la carte a laissé le Neural Engine éteint, le tag est un simple fp32 et il n'y a rien à choisir.

Sur le M1 Pro, 1024², Standard, rendu plus rapide, graine 42, le prompt de la bibliothèque, le 23/10/2026 :

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo |
|---|---|---|---|
| temps dans l'app, éteint / allumé | 144,6–144,9 s / 135,4–137,4 s | 86,8–93,9 s / 83,4–84,0 s | 103,6 s / 99,9–100,2 s |
| puissance de toute la machine, éteint / allumé | 43,9 W / 35,3 W | 38,0 W / 31,2 W | 45,4 W / 36,9 W |
| énergie par image, allumé contre éteint | −26 % | −16 % | −20 % |
| pic mémoire de l'app, éteint / allumé | 3,41 / 3,71 Go | 3,01 / 3,23 Go | 1,97 / 2,11 Go |
| l'image, allumé contre éteint | PSNR 35,9 dB : la même scène, le même visage, la même lumière, les détails les plus fins déplacés (tranches de livres) | 55,9 dB, un dixième de niveau en moyenne | 56,3 dB, un dixième de niveau en moyenne |
| écart de la sortie du débruiteur à l'exact fp64, la moitié des lignes de chaque produit sur le Neural Engine | 4,8·10⁻³ (512²) | 3,6·10⁻³ (1024²) | 4,4·10⁻³ (512²) |
| …en part de l'écart du bf16 de l'éditeur | 0,14 | 0,06 | 0,31 |

Temps : des rendus mis en file dans l'app par `silicontrol`, chacun dans une app tout juste ouverte, 0 page swappée ; éteint, trois rendus de Z-Image d'affilée ont pris 93,9, 89,0 et 86,8 s aux mêmes bits, le premier juste après un déplacement de la bibliothèque. Puissance et énergie : la même comparaison en ligne de commande, quatre rendus par modèle en alternant allumé et éteint, la consommation de toute la machine (écran compris) lue dans la télémétrie de l'adaptateur secteur. L'écart est celui d'une évaluation du débruiteur avec plus de Neural Engine qu'aucune carte n'en donne, la moitié des lignes de chaque produit, tel que l'imprime `siliconed check <modèle> dit --neural-engine` ([VERIFY.fr.md](VERIFY.fr.md)), le 24/10/2026 ; celui du bf16 est celui de diffusers sur les poids bf16 de l'éditeur, comme calculent la plupart des outils.

## Pourquoi Siliconed

- **Vérifié contre les modèles d'origine.** Chaque étage (encodeur de texte, débruitage, VAE) est
  comparé, canal par canal, à l'implémentation de référence (diffusers / transformers en fp32 sur
  CPU), et depuis la 0.12 arrondit là où la référence arrondit. Ce qui ne se vérifie pas contre la
  référence n'est pas livré.
- **Tout en fp32, et rapide quand même.** Z-Image Turbo à 1024² en 84,5 s sur un M1 Pro avec les
  8 évaluations que demande sa recette, les trois premières sur une grille en demi-taille
  ([rendu plus rapide](#rendu-plus-rapide), allumé par défaut) ; 117,7 s toutes en pleine taille
  ([à côté de Draw Things](#à-côté-de-draw-things), la référence sur Mac). Là où la carte de ce Mac
  le permet, une partie du débruitage passe sur le Neural Engine, pour 16 à 26 % d'énergie en moins
  par image sur un M1 Pro : à votre choix sur la Standard, par défaut sur la Compact et la Légère
  ([comment](#le-neural-engine)).
- **Ajusté à ce Mac.** Aucun noyau ne s'allume d'après le nom de la puce : la carte mesurée sur ce
  Mac allume ou éteint chacun ([ce qu'elle a décidé sur un M1 Pro](#un-moteur-mesuré-sur-votre-mac)).
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
- **Reproductible au bit près.** Même prompt, même graine, mêmes réglages, même version, sur le même
  Mac : la même image, octet pour octet. Une version peut changer les pixels : la 0.12 le fait, de
  quelques unités sur le dernier chiffre, maintenant que son arithmétique arrondit comme la référence.

## L'app

`Siliconed.app` génère des images à partir d'un prompt, les édite par instruction, et tient une file
et un historique. Elle suit la langue du système (anglais, français, allemand, espagnol ou italien).

![La fenêtre : le rack à gauche (prompt, encodeur de texte, débruiteur, décodeur, chacun avec ses réglages et son temps), une image Krea 2 Turbo en 1248×832 à droite — une Irlandaise rousse qui sort des vagues en riant, sa planche de surf sous le bras, sous un ciel ensoleillé —, l’historique en dessous](docs/images/window.jpg)

*La fenêtre, avec la photo de Krea 2 Turbo plus haut : le rack montre la chaîne, les réglages et le
temps de chaque étage ; l'historique court sous le canevas.*

### Créer

- **Quatre modèles.** Krea 2 Turbo, Z-Image Turbo, Qwen-Image-2.1 Turbo et Anima Turbo, chacun sous
  sa propre licence ([Les modèles](#les-modèles)). Tous quatre lisent le prompt avec un modèle de
  langage.
- **La chaîne, visible.** Le rack montre le pipeline de haut en bas (prompt, encodeurs, débruitage,
  décodeur), avec des fils typés et l'étage actif allumé ; chaque étage porte ses propres réglages.
- **Plus de détail** avec Z-Image et Qwen-Image-2.1 : « Détail : Normal · Plus · Maximum » ajoute
  de la texture fine, sans temps en plus.
- **Rendu plus rapide** (Réglages, allumé par défaut) : dès 1024 px de chaque côté,
  les premières évaluations sur une grille en demi-taille, le reste en pleine taille : d'un quart à
  deux cinquièmes plus rapide, toujours 8 évaluations, une image un peu différente
  ([comment](#rendu-plus-rapide)).
- **Les variations** : des cousines d'une image, subtiles ou marquées, aux mêmes réglages.
- **Agrandir ×4** avec la super-résolution de macOS (macOS 26) : aucun prompt, rien d'ajouté, bords
  et textures affinés ; 1024² devient 4096² en une ou deux secondes.
- **Les LoRA** sur les quatre modèles.

### Éditer

- **L'édition par instruction** avec Qwen-Image-2.1 : donnez-lui 1 à 3 images et une phrase
  (« remplace le ciel nuageux par un ciel bleu ») ; son encodeur voit les images. Dans l'app, une
  édition est plafonnée à une surface de 1024².
- **Le rideau** : faites glisser une ligne sur deux images, ou sur une édition ou un agrandissement
  et son original, pour voir ce qui a changé ; zoomez vers le curseur pour regarder un détail.
- **Un historique comme dans Photos** : flèches, ⌫ et ⌘Z, sélection multiple, comparaison côte à
  côte, export, glisser-déposer, Partager.

![Une édition par Qwen-Image-2.1 dans la fenêtre : la photo Krea 2 de la surfeuse en image 1 dans la boîte Images, l’instruction « Convert this photo to a western comic book style with cel shading », le résultat en cel shading dans le canevas](docs/images/edit.jpg)

*Qwen-Image-2.1 Turbo, Standard : « Convert this photo to a western comic book style with cel shading », la photo Krea 2 en image 1, 1248×832, graine 42, 215 s.*

![Une édition par Qwen-Image-2.1, sous le rideau qui glisse : la surfeuse de Krea 2 qui sort des vagues en combinaison, puis en chemisier de lin blanc, son visage, ses cheveux, sa planche et la mer inchangés](docs/images/outfit.gif)

*Une édition qui ne change que ce qu’on lui dit : « Replace her wetsuit with a white linen blouse. Keep her face, hair, surfboard, pose and the beach unchanged. », la photo Krea 2 en image 1, 203 s.*

### Explorer

Le panneau d'exploration construit une grille dans le canevas. Ses axes : les graines, les pas, la
force d'une LoRA, des LoRA à essayer, les modèles, les variantes écrites entre accolades dans le
prompt, chaque image d'une édition ou, en mode développeur, les solveurs. Chaque case est une
esquisse, arrêtée après quelques pas, à 512 sur le petit côté (avec Qwen-Image-2.1, une image
finie) ; dès qu'un modèle a rendu à cette taille, le panneau dit ce que la grille coûtera sur ce Mac
avant de la lancer. « Finir cette image » rend la case choisie jusqu'au bout avec la même graine :
l'image que son esquisse annonçait.

![Une exploration avec Anima Turbo et la LoRA inCase qui se remplit dans le canevas : des esquisses d’une Irlandaise rousse qui rit sur une plage ensoleillée au pied des falaises, la couleur de la combinaison en colonnes (teal, coral red, mustard yellow, navy) et la force de la LoRA en lignes (0 ; 0,5 ; 1), les axes en étiquettes, la dernière ligne encore en cours](docs/images/grid.jpg)

*Une exploration avec Anima Turbo et la LoRA inCase : la couleur de la combinaison écrite entre
accolades dans le prompt (`{teal|coral red|mustard yellow|navy}`) en colonnes, inCase à 0, 0,5 et 1
en lignes ; des esquisses en 512×768, environ 16 s chacune sur un M1 Pro.*

![La case finie dans la fenêtre : une illustration d’Anima Turbo avec la LoRA inCase, une jeune Irlandaise rousse qui rit, en combinaison bleu marine, en gros plan, des falaises et des vagues qui déferlent derrière elle](docs/images/anima.jpg)

*La case choisie, finie : Anima Turbo avec la LoRA inCase à 1, « A close-up portrait of a young
Irish woman with long wavy auburn red hair, pale freckled skin and green eyes, laughing, wearing a
navy shorty wetsuit, on a sunny cliff-top beach in the west of Ireland. Big Atlantic waves breaking
into white spray below green sea cliffs, deep blue sky, bright sunlight, wind in her hair. »,
512×768, graine 4001, 20,8 s.*

### La gestion des modèles

- **Rien n'est préinstallé.** Vous choisissez un modèle ; avant le premier octet téléchargé, l'app
  affiche sa licence et la place qu'il prend sur disque, et vous demande d'accepter. Chaque fichier
  est figé à une révision et vérifié par sha256.
- **Standard, Compact ou Légère.** Chaque modèle s'installe avec les poids de l'éditeur (Standard,
  le défaut), avec des poids 8 bits publiés par des tiers (Compact : 19,5 Go au lieu de 34,4 pour
  Krea 2, 11,7 Go au lieu de 20,4 pour Z-Image, 17,3 Go au lieu de 31,2 pour Qwen-Image-2.1, 4,10 Go
  au lieu de 5,65 pour Anima ; une image légèrement différente), ou avec le transformeur en 4 à 6 bits
  tel qu'un tiers le publie (GGUF Q4_K_M ; Légère : 12,8 Go, 9,5 Go, 14,2 Go et 3,31 Go, le moins lu
  sur le disque à chaque pas ; un autre jeu de poids, une image qui peut se composer autrement). Pour
  Krea 2, Z-Image et Qwen-Image-2.1, l'encodeur de texte de la Compact est lui aussi en 8 bits, et la
  Légère le reprend ; celui d'Anima reste le bf16 de l'éditeur dans les trois, faute de version 8
  bits publiée. La Légère est présélectionnée sur un Mac de 8 Go de mémoire, la Standard ailleurs ;
  vous choisissez avant d'installer. Une version à la fois : passer à l'autre la remplace ;
  l'installée continue de rendre jusqu'à ce que la nouvelle soit complète, puis libère sa place ; si
  le disque ne peut pas tenir les deux, l'app le dit avant l'acceptation.
- **Les imports Civitai et Hugging Face.** Les LoRA (formats Civitai, Comfy, kohya, diffusers) et le
  DiT d'un fine-tune (`.safetensors`, ou `.gguf` de Q4 à Q8_0), par glisser-déposer ou ⌘O ; un
  checkpoint Civitai d'une famille connue s'importe tel quel et emprunte l'encodeur et le VAE de sa
  famille. Déposé avant que sa famille soit installée, il ouvre l'installation de celle-ci — licence,
  encodeur et VAE, jamais le DiT de la famille — et s'importe une fois l'installation faite. Un
  fichier quantifié reste sur le disque tel que publié, jamais requantifié (Z-Image : 6,2 Go en
  8 bits, environ 5 Go en GGUF Q4_K_M, au lieu de 12,3 Go ; chaque pas un peu plus lent, le calcul
  reste en fp32) ; sous 4 bits, le fichier est refusé. Une LoRA importée prend le nom de son
  fichier ; le nom donné par son auteur et ses mots déclencheurs s'affichent en infobulle.
- **Statistiques et diagnostic.** La palette des statistiques (⌥⌘I) est un bouton de la barre
  d'outils. « Mesurer ce Mac » (menu Aide, ou « Chaîne » en tête du rack, orange tant que
  ce Mac n'est pas mesuré) montre la carte de ce Mac : ce que calcule la puce, mesuré, et les chemins
  que prend le moteur.

Les modèles vivent dans `~/Library/Application Support/Siliconed` (modifiable avec `SILICONED_ROOT`). Pour
libérer le disque interne, les Réglages les déplacent sur un autre — un disque externe, un partage
réseau : chaque fichier est copié, vérifié, puis effacé du Mac. Siliconed ne copie alors sur le Mac que
les modèles que vous rendez, le temps de la session, et les efface en quittant ; les licences acceptées
et la mesure de ce Mac restent sur le Mac. Débranché, le disque est signalé : rien ne se rend avant
son retour.

![La fenêtre Modèles : les quatre modèles des éditeurs installés en Standard, chacun avec sa licence et sa place sur le disque (Z-Image Turbo 20,36 Go, Qwen-Image-2.1 Turbo 31,21 Go, Anima Turbo 5,65 Go, Krea 2 Turbo 34,4 Go), la LoRA inCase pour Anima en dessous, la place occupée et libre en bas](docs/images/models.jpg)

### Privé par construction

Le rendu se fait sur votre Mac. Le seul trafic réseau est le téléchargement du modèle que vous avez
accepté ; le diagnostic copie son rapport et ouvre un formulaire d'issue GitHub que vous choisissez
de soumettre ou non. Le cache de rendu ne dure qu'une session : il est vidé au lancement et à la
fermeture. Avant chaque rendu, le moteur vérifie le modèle, sa licence, le format et la mémoire qu'il
lui faut, et refuse d'emblée un rendu au-delà du budget de ce Mac. L'app ne censure aucun prompt et n'ajoute aucun filtre de contenu ; la
licence de chaque modèle s'applique toujours, et celle de Krea 2 Community en demande un à qui déploie le
modèle.

## Les modèles

Seulement des modèles récents dont l'encodeur de texte est un modèle de langage — ni Stable
Diffusion, ni SDXL, ni Flux.1.

| Modèle | Éditeur | Ce qu'il fait | Licence |
|---|---|---|---|
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea | texte vers image (des images esthétiques, de la photographie à la peinture), LoRA | Krea 2 Community : usage commercial sous 1 M$ de chiffre d'affaires annuel, un filtre de contenu obligatoire au déploiement |
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | texte vers image, LoRA | Apache 2.0 |
| [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Qwen | texte vers image, édition par instruction avec 1 à 3 images, LoRA | Qwen Research (non commerciale) |
| [Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima) | CircleStone Labs × Comfy Org | texte vers image (illustration, peinture, anime ; pas de photoréalisme, par choix de l'éditeur), LoRA | CircleStone Labs Non-Commercial ; ses images sont utilisables commercialement |

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
avec l'encodeur de texte de la Compact.

Les autres versions Compact utilisent des poids 8 bits publiés par des tiers :
[unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF),
[Disty0/Z-Image-Turbo-SDNQ-int8](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-int8),
[unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8),
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) ; les versions Légères, le
GGUF Q4_K_M de [unsloth/Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) et
d'[unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) avec l'encodeur de
texte de la Compact. La Compact et la Légère d'Anima prennent le DiT en GGUF Q8_0 et Q4_K_M de
[vanes430/Anima-Turbo-V1.1-GGUF](https://huggingface.co/vanes430/Anima-Turbo-V1.1-GGUF) et gardent
l'encodeur de texte de l'éditeur en bf16 (aucune version 8 bits n'en est publiée). Chaque fichier est
figé à une révision et vérifié par sha256 (`Sources/Siliconed/Forge/Installation.swift`).

## Télécharger

L'app est distribuée en `.dmg` sur la page
[Releases](https://github.com/gwenn-ha-dev/Siliconed/releases). Prérequis : macOS 15 ou plus, Apple
silicon. Les temps ci-dessous ont été mesurés avec 16 Go de mémoire.

## Les chiffres

Machine de référence : MacBook Pro M1 Pro (8 cœurs, 14 cœurs GPU), 16 Go. Graine 42, 8 pas.

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo |
|---|---|---|---|---|
| 1024², Standard, rendu plus rapide (par défaut) | **145,6 s** | **84,5 s** | **103,7 s** | **32,7 s** |
| 1024², Standard, toutes les évaluations en pleine taille | 221,1 s | 117,7 s | 142,7 s | 44,4 s |
| 1024×1536, Standard, rendu plus rapide / pleine taille | 216,9 s / 355,3 s | 131,0 s / 190 s | 161,7 s / 232 s | 52,5 s / 75,6 s |
| autres, Standard | | | une édition à 1 image 211 s, à 2 images 297 s | 512² 12,1 s |
| 1024², Légère (4 à 6 bits), rendu plus rapide (par défaut) | 153,8 s | 89,7–91,1 s | 111,0 s | 35,4 s |
| pic mémoire, Standard / Légère | 3,07 Go (1024×1536 : 3,51 Go), le moteur seul / 1,55 Go | 2,8 Go / 2,1 Go | 1,8–2,7 Go / 1,5 Go | 1,4 Go (le moteur seul) / 1,4 Go |
| écart de la sortie du débruiteur (`model_out`, pire canal) à la référence fp32 · à l'exact fp64 (celui de la référence) | 2,6·10⁻⁴ · 3,3·10⁻⁴ (3,1·10⁻⁴), 1024² | 9,8·10⁻⁶ · 1,2·10⁻⁵ (1,3·10⁻⁵), 1024² | 3,1·10⁻⁶ · 2,6·10⁻⁶ (2,5·10⁻⁶), 512² | 1,4·10⁻⁵ · par plan latent 2,9·10⁻⁶ (3,8·10⁻⁶), 1024² |

Chaque temps est celui d'un rendu mis en file dans l'app par `silicontrol`, de la file au PNG, la
carte de ce Mac en service. Avec le rendu plus rapide : le 19/10/2026, app 0.12, le réglage à sa
valeur par défaut ; un rendu de Z-Image en 1024² réglage éteint, en témoin, a pris 112,0 s dans la
même série. La Légère, et le pic de Z-Image en Standard : le 19/10/2026 au soir, app 0.12, le réglage à
sa valeur par défaut, chaque rendu dans une app tout juste ouverte, 0 page swappée ; le premier et le
dernier rendu de Z-Image en Légère, en témoins, ont pris 91,1 et 90,5 s. Toutes les évaluations en pleine taille, 1024² : le 18/10/2026, app 0.12, 0 page
swappée ; un dernier rendu de Z-Image en fin de série, en témoin, a pris 123,6 s (la machine dérive
de quelques pour cent d'un rendu à l'autre) ; le 1024×1536 de Krea 2 : la même série. Les autres temps :
Qwen-Image-2.1 le 15/10/2026 avec la 0.10, Z-Image le 14/10/2026 avec la 0.8.2. Le pic mémoire est
le `phys_footprint` de l'app, relevé toutes les 100 ms. Les autres temps et le pic d'Anima Turbo sont
ceux du moteur seul, sans l'app, cache de rendu éteint (14–15/10/2026) ; les pics de Krea 2 Turbo
en Standard aussi (17/10/2026). Les écarts sont ceux qu'a imprimés
`siliconed check <modèle> dit --size <côté>` sur le même Mac le 24/10/2026, contre des goldens écrits par
`tools/oracle/` ; chez vous, ils diffèrent sur les derniers chiffres ([VERIFY.fr.md](VERIFY.fr.md), [La vérification](#la-vérification)).

Jusqu'à la 0.8.1, Z-Image Turbo faisait 7 des 8 évaluations du DiT que demande sa recette ; les temps
publiés alors (82–85 s à 1024²) valaient pour 7. Corrigé dans la 0.8.2 : 8 évaluations en pleine
taille. Le rendu plus rapide n'est pas ce raccourci : il garde les 8 évaluations.

### Rendu plus rapide

Allumé par défaut, en tête des Réglages, pour les quatre modèles. Dès 1024 px de chaque côté (1024²,
1024×1536, 1536×1024…), les premières évaluations se font sur la grille en demi-taille — 3 sur 8 pour Z-Image,
Qwen-Image-2.1 et Anima, 4 pour Krea 2, dont le calendrier garde ses premiers niveaux de bruit plus
hauts — et le reste en pleine taille. Toujours 8 évaluations ; une évaluation en demi-taille coûte
environ un quart d'une pleine. Aucun format portrait ou paysage usuel n'en profite : la moitié de
832×1216 passerait sous 512 px d'un côté.

La petite grille reçoit l'état moyenné plus la basse bande du bruit de départ : elle tient une image
propre, bruitée comme il faut ; la vitesse ramenée en pleine taille est celle qui mène l'état complet
à l'estimation propre agrandie : elle emporte aussi le bruit fin qu'une demi-grille ne voit pas, au
rythme d'un pas plein. Les pas en demi-taille retirés dans la 0.8.2 n'avaient ni l'un ni l'autre :
leur vitesse agrandie laissait ce bruit fin en place (du sable sur la peau et les murs), et leur
petite grille était une simple décimation, dont le repliement se lisait comme une texture (du
craquelé). Le nombre d'évaluations en demi-taille de chaque modèle a été jugé à l'œil sur deux prompts, un portrait et la
bibliothèque : ni sable ni craquelé, taches de rousseur, cheveux et dos
des livres intacts ; un pas de plus, et le détail fin commence à partir.

Il donne une autre image que la recette en pleine taille à la même graine (la composition se décide
dans les premiers pas : la même scène, parfois une autre personne), pas une copie dégradée. Pas avec
une LoRA, Détail, une édition, un autre nombre de pas ni un autre calendrier. Éteint, toutes les
évaluations sont en pleine taille et l'image est la même, au bit près, qu'avant : celle qui est
vérifiée contre diffusers ; les écarts ci-dessus valent pour elle.

### À côté de Draw Things

[Draw Things](https://drawthings.ai) est la référence sur Mac ; ici, son moteur communautaire,
[draw-things-community](https://github.com/drawthingsai/draw-things-community) à `bc3970c`, avec
les réglages par défaut du modèle (UniPC trailing, CFG 1). La même machine, tout juste redémarrée, un
seul moteur chargé à la fois ; Z-Image Turbo, 1024², 8 pas (8 évaluations de chaque côté), graine 42,
trois rendus de chaque côté sur trois variantes du même prompt. Draw Things a été mesuré le 13/10/2026
(une séance de 70 minutes) et rien n'a changé de son côté depuis ; Siliconed en pleine taille le
14/10/2026, après le correctif, et avec le rendu plus rapide le 19/10/2026 (app 0.12, trois rendus
du prompt de la bibliothèque chacun, chacun dans une app tout juste ouverte).

| | poids | temps | pic mémoire | lu sur le disque par rendu |
|---|---|---|---|---|
| Siliconed, Standard | les bf16 de l'éditeur, calculés en fp32 | 113–119 s | 2,6 Go | 51–58 Go |
| Siliconed, Légère | GGUF Q4_K_M | 118–120 s | 2,0 Go | 3,5–8,7 Go |
| Siliconed, Standard, rendu plus rapide (par défaut) | les bf16 de l'éditeur, calculés en fp32 | 85–86 s | 2,8 Go | 51–58 Go |
| Siliconed, Légère, rendu plus rapide (par défaut) | GGUF Q4_K_M | 90–91 s | 2,1 Go | 3,5–8,7 Go |
| Draw Things, Exact | f16 | 167–179 s | 5,3 Go | 99 Go |
| Draw Things, par défaut | 8 bits (`q8p`) | 142–164 s | 3,5–4,3 Go | 11 Go |
| Draw Things, 6 bits | 6 bits (`q6p`) | 136–138 s | 3,5 Go | 6–14 Go |

La comparaison porte sur le temps écoulé, pas sur une arithmétique identique : les solveurs et les
poids diffèrent, les images aussi. Le pic mémoire ne compte pas les poids projetés depuis le disque,
d'un côté comme de l'autre. Siliconed fait chauffer davantage la machine, ce qui découle du calcul
en fp32. Jusqu'à la 0.8.1, ce tableau opposait les 7 évaluations de Siliconed aux 8 de Draw Things ;
il compare maintenant 8 à 8. Les deux premières lignes de Siliconed sont en pleine taille, comme
celles de Draw Things ; avec le rendu plus rapide, les 3 premières de ses 8 évaluations passent sur
une grille de demi-taille, ce que Draw Things ne fait pas : moins de travail, une autre image
([comment](#rendu-plus-rapide)).

Huit bits contre huit bits, le 23/10/2026 : la Compact de Siliconed (le GGUF Q8_0 du tiers, gardé tel que publié) contre le `q8p` par défaut de Draw Things, dans son moteur communautaire à `6c1dd09` ; Z-Image Turbo, 1024², 8 pas, graine 42, le prompt de la bibliothèque ; chaque rendu dans un processus tout juste lancé de chaque côté (l'app, le serveur du moteur), trois de chaque côté, puis Siliconed encore en témoin.

| | poids | temps | pic mémoire | lu sur le disque par rendu |
|---|---|---|---|---|
| Siliconed, Compact, rendu plus rapide et Neural Engine (tous deux par défaut) | GGUF Q8_0 | 81,1–83,2 s (témoin 81,3 s) | 2,6 Go | 3,7–11 Go |
| Draw Things, par défaut | 8 bits (`q8p`) | 135,0–144,2 s | 4,3 Go | 10–18 Go |

**Dans le budget, mesuré.** Le pic `phys_footprint` de chaque modèle et ses pages swappées (delta
`vm_stat`) sont mesurés au pire cas du produit, sur un Mac de 16 Go où rien d'autre ne tourne : Qwen-Image-2.1 à 1024×1536 avec 3 références,
0 page swappée ; Z-Image à 1024×1536, pic 3,8 Go, 0 page swappée (son VAE décode en bandes, aux
mêmes bits).

Chaque côté doit être d'au moins 512 et multiple de 16, pour une surface d'au plus 1024×1536. Sous
512² les modèles sont hors de leur domaine. Qwen-Image-2.1 veut des multiples de 32 (son VAE divise
par 16), et Anima au plus 2048 px de côté (les positions que sa référence calcule) ; l'un et l'autre
sont refusés avant tout calcul.

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

Une machine ne fait pas un banc. **Mesurer ce Mac** (menu Aide ; *Measure This Mac*
en anglais) montre la carte que ce Mac a mesurée avant son premier rendu, en une minute environ et
sans aucun modèle : les produits matriciels des blocs des modèles sur le GPU, sur l'unité
matricielle du CPU, sur le Neural Engine et sur le GPU à côté de chacun, des produits fp16 et bf16, des produits sur les
poids int8 et Q8_0 publiés, l'attention par les trois noyaux du moteur, l'élargissement des poids sur
le CPU et sur le GPU — chaque résultat vérifié contre le résultat exact calculé en double précision —
et ce que la carte en a décidé. Il copie le rapport JSON et ouvre un formulaire d'issue GitHub où le
coller (ou vous l'envoyez autrement) : puce, cœurs CPU et GPU, mémoire, macOS, drapeaux du CPU,
chaque débit et s'il a calculé juste, état thermique et alimentation avant et après. Rien n'est envoyé
tant que vous ne soumettez pas l'issue.

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

macOS 15 ou plus, Apple silicon, Xcode 26 ou plus (l'app emploie les API de macOS 26 là où le système les a).

```
swift build                 l'app et silicontrol
swift test                  les tests unitaires purs, sans aucun poids
tools/app.sh                construit Siliconed.app à la racine du dépôt et l'ouvre
tools/app.sh --no-open      construit seulement
tools/dmg.sh                emballe l'app construite en .build/Siliconed-<version>.dmg
```

`swift test` ne demande rien d'autre : ni poids, ni Python. Siliconed livre une app, pas une
bibliothèque : les déclarations du moteur restent internes au paquet.

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
dérive de plusieurs pour cent d'un rendu à l'autre. La comparaison à la référence est livrée : les
scripts d'oracle de `tools/oracle/` et la commande `siliconed check` la refont sur votre Mac, étape
par étape, avec les empreintes de nos goldens pour y comparer les vôtres ; [VERIFY.fr.md](VERIFY.fr.md)
dit comment, et ce que ça coûte. Le journal des mesures ne fait pas partie de ce dépôt. Le diagnostic
de l'app vérifie une machine contre des résultats exacts calculés sur son propre CPU.

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
