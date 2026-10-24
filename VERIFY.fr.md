[English](VERIFY.md) · Français

# Vérifier Siliconed

Siliconed affirme que chaque étage de chaque modèle qu'il livre calcule ce que calcule l'implémentation de référence. **Cette page montre comment le vérifier soi-même**, sur son Mac, sans nous croire sur parole.

La référence est celle de l'éditeur : [diffusers](https://github.com/huggingface/diffusers) et [transformers](https://github.com/huggingface/transformers), en float32 sur le CPU, avec les poids publiés. Des scripts Python l'exécutent et figent ses réponses en *goldens* ; `siliconed check` fait tourner le moteur sur les mêmes entrées et le juge contre elles, étage par étage.

## Cinq minutes, sans poids

Les noyaux numériques (tables RoPE, rééchantillonnage, opérations élément par élément du GPU, attention, partage d'un produit matriciel entre GPU et AMX) se vérifient sans aucun modèle.

`siliconed` est livré dans l'app (0.15 et suivantes) : c'est le moteur même avec lequel l'app rend, et il tourne sans Xcode. Les scripts d'oracle viennent de ce dépôt, à la version de l'app (Siliconed → À propos l'affiche) :

```sh
alias siliconed=/Applications/Siliconed.app/Contents/Helpers/siliconed
git clone --branch v0.15 https://github.com/gwenn-ha-dev/Siliconed && cd Siliconed
python3.12 -m venv oracle-env && oracle-env/bin/pip install -r tools/oracle/requirements.txt
oracle-env/bin/python tools/oracle/kernels.py all          # les goldens RoPE et rééchantillonnage, quelques secondes
siliconed check kernels
```

Pour vérifier plutôt votre propre compilation : `swift build -c release --product siliconed` (Xcode 26), puis `.build/release/siliconed` à la place de l'alias.

Chaque étape affiche ses mesures, puis une ligne de verdict, `✓`, `✗` ou `—` (sautée, avec la raison). Le code de sortie dit la même chose : 0 réussi, 1 échoué, 2 sauté, 64 erreur d'usage.

## Une étape de modèle

Installer d'abord un modèle dans l'app (Z-Image est le plus léger à télécharger) : `siliconed check` lit les poids que l'app a installés, jamais ceux que télécharge l'oracle. Puis, pour l'encodeur de texte :

```sh
oracle-env/bin/python tools/oracle/z-image.py text         # télécharge une fois les poids de référence
siliconed check z-image text
```

`siliconed check list` liste chaque étape, ce qu'elle prouve, comment elle est jugée et quelle commande écrit ses goldens ; la table en bas de page est sa sortie. Les goldens vont dans `./goldens`, sauf si `--goldens DIR` ou `SILICONED_GOLDENS` en dit autrement, pour les scripts comme pour les contrôles. Pour écrire d'un coup tous les goldens d'un modèle : `oracle-env/bin/python tools/oracle/z-image.py all` (de même `qwen-image-2.1.py`, `anima.py`, `krea2.py`) ; ce que ça coûte est plus bas.

## Vos goldens sont-ils les nôtres ?

Nous ne publions pas les goldens (plusieurs gigaoctets, et des poids dont les licences l'interdisent). Nous publions leurs empreintes : forme, type et statistiques de chaque tenseur.

```sh
oracle-env/bin/python tools/oracle/fingerprints.py compare
```

Les entiers (identifiants de tokens, masques) doivent concorder exactement. Les flottants se comparent avec une tolérance (1e-3 sur les normes), pas au bit près : PyTorch sur le CPU ne promet pas les mêmes bits float32 d'une machine à l'autre, et deux exécutions honnêtes de la référence diffèrent d'un arrondi. Cet arrondi est très en dessous de ce que jugent les contrôles.

## La règle

Le chiffre d'une étape est l'**écart relatif par canal** : pour chaque canal (le dernier axe d'un tenseur, ou chaque plan d'un latent), la norme L2 de la différence sur la norme L2 de la référence. On donne le pire canal, jamais une moyenne : une faute confinée à un canal disparaîtrait dans un chiffre global. La même règle juge chaque frontière de chaque modèle :

    sans témoin :   écart(moteur, référence)  ≤ 5e-4
    avec témoin :   écart(moteur, témoin)     ≤ max(plancher, 1,25 · écart(référence, témoin))

Le *témoin* est le même calcul en float64. Là où il existe, il tranche : le float32 a sa propre erreur d'arrondi, et sur une frontière mal conditionnée la référence elle-même peut être à 4e-3 de la valeur exacte ; le moteur peut alors en être aussi loin que la référence, plus 25 %. La ligne de chaque étape dans `check list` donne sa règle : une identité (tokens, plannings) se juge au bit près, quelques étapes ont un plancher plus serré, et un rectangle se contrôle aussi contre sa transposée, qui doit échouer, pour qu'une hauteur et une largeur inversées ne passent pas.

Le débruiteur de Krea 2 a quelques frontières si mal conditionnées qu'un changement d'une unité sur le dernier chiffre de l'entrée déplace la sortie de plus que la règle ne permet, référence comprise. Là, une mesure qui échoue au témoin est jugée par *nuages* : la référence et le moteur tournent tous deux sur des entrées décalées d'un ulp, et l'écart médian du moteur doit rester sous 1,25 fois le pire de la référence. Ce sont les étapes dont la règle dit « clouds when the oracle has draws ».

Deux mesures de Krea 2 passent de peu sur notre M1 Pro : la vitesse `v_2` de la trajectoire en 512² (5,08e-4 pour une tolérance de 5,14e-4) et `block0_attn_out` du débruiteur en 1024² (3,53e-3 pour 3,59e-3). La tolérance se calcule sur l'écart de votre propre référence, elle suit donc les arrondis de votre processeur, et ces deux mesures peuvent échouer d'un cheveu sur un autre Mac. La règle n'est pas desserrée pour elles ; si elles échouent chez vous, une issue avec la sortie de `siliconed check` nous aide à trouver la couche qui porte l'écart.

**Le Neural Engine.** L'app peut partager les produits matriciels du débruiteur avec le Neural Engine du Mac, qui calcule en fp16 ([README](README.fr.md#le-neural-engine)). `--neural-engine` vérifie ce chemin : chaque produit élargi du DiT donne la moitié de ses lignes à l'ANE, le même partage sur tous les Mac et plus qu'aucune carte n'en donne, et l'étape est jugée contre la référence calculée en bf16 sur les mêmes entrées, le moteur ayant droit à l'écart propre au bf16 plus 25 %. Ce calcul bf16 est une étape de script de plus, hors de `all` :

```sh
oracle-env/bin/python tools/oracle/z-image.py bf16 --size 1024
siliconed check z-image dit --size 1024 --neural-engine
```

Le verdict dit combien de produits l'ANE a calculés ; sur un Mac où il n'en calcule aucun, l'étape est sautée. Les étapes `dit` des quatre modèles l'acceptent.

Ce qui n'est **pas** affirmé : les mêmes bits que la référence, ni qu'un autre Mac. Le moteur calcule dans un autre ordre sur un autre processeur ; il est jugé sur sa distance à la valeur exacte, pas sur ses bits. Les LoRA ne sont pas couvertes ici : ce sont des fichiers que vous choisissez, pas une part de ce que Siliconed livre.

## Ce que ça coûte

Sur un MacBook Pro M1 Pro, 16 Go. Téléchargement : les poids de référence que récupèrent les scripts, en plus de ceux de l'app. Oracle : le `all` des scripts, temps écoulé, puis son étape la plus lourde avec son pic mémoire. Contrôle : `siliconed check` sur toutes les étapes du modèle, temps écoulé.

| Modèle | Téléchargement | Oracle, toutes les étapes | Son étape la plus lourde | Contrôle |
|---|---|---|---|---|
| noyaux | — | moins de 5 s | — | quelques secondes |
| Z-Image | 31 Go | 39 min | la trajectoire 1024² : 14 min, 13,3 Go | 7 min |
| Qwen-Image-2.1 | 30 Go | 27 min | le témoin fp64 du décodeur : 43 s, 15,5 Go | 10 min |
| Anima | 5 Go | 7 min | le décodeur 1024² : 1 min, 20,8 Go | 3 min |
| Krea 2 | 33 Go | 1 h 45 min | les huit témoins fp64 des pas à 1024² : 44 min, 11 Go | 16 min |

L'oracle fait passer le débruiteur bloc par bloc en float32 (celui de Krea 2 pèse 48 Go à cette précision), si bien qu'un Mac de 16 Go exécute chaque script ; une étape dont le pic dépasse la mémoire de la machine (les décodeurs 1024²) tourne en swap et prend plus longtemps.

## Les étapes

La sortie de `siliconed check list --markdown`, telle que la commande l'écrit, en anglais. `{size}` vaut 512, 1024 ou un rectangle comme 768x512 ; les étapes sans taille se contrôlent à celle de la référence.

| Stage | Proves | Rule | Oracle | Weights |
|---|---|---|---|---|
| `kernels widen` | bf16 → fp32 widening and its inverse are exact, on the values a naive shift gets wrong | bit for bit | — | — |
| `kernels arena` | the arena hands out distinct page-aligned slices, refuses duplicates and growth | each property holds | — | — |
| `kernels elementwise` | the GPU's SwiGLU, gate and gated residual are as close to double precision as the CPU's; the residual rounds twice, as torch | GPU worst ulp ≤ max(CPU's, 1), mean ≤ CPU's + 5 %; residual bit for bit | — | — |
| `kernels gelu` | the GPU's GELU (erf) is as close to double precision as the CPU's, over one bit pattern in 64 of [−12, 12] | GPU worst deviation ≤ the CPU's, in ulp of the operand | — | — |
| `kernels sdpa` | the attention cut by heads (GPU + CPU), by query blocks, past maxMatrix and the matrix kernel return the whole graph's result | against the whole graph: query blocks ≤ 1e-6 per channel; head cut ≤ 1e-5 with the GPU heads bit for bit; matrix kernel ≤ 1e-5 | — | — |
| `kernels gemm-split` | the token-wise split of a GEMM between the GPU and the AMX stitches back cblas_sgemm's product | against cblas_sgemm, per row: the GPU's rows bit for bit, the AMX's ≤ 1e-5 (another summation order over k = 3 840) | — | — |
| `kernels rope` | every RoPE table of the engine is the reference's, value by value (13 tables, 5 bit for bit) | max \|Δ\| per table within its bound, in units of 2⁻²⁴: 0 where the reference is bit-reproducible, 1 to 64 where torch uses SLEEF's cos, sin, pow | `tools/oracle/kernels.py rope` | — |
| `kernels resample` | the spectral schedule's nearest-exact decimation and bilinear upsampling are torch's, on a square and a rectangle | decimation bit for bit (and the even decimation, its counter-test, must differ); bilinear within one ulp of the input's scale | `tools/oracle/kernels.py resample` | — |
| `z-image tokenizer` | the chat template and the tokenizer give transformers' 512 identifiers and mask, padding included | bit for bit | `tools/oracle/z-image.py text` | tokenizer |
| `z-image text` | the Qwen3-4B encoder gives the reference's hidden states, layer by layer, and its cap_feats | the rule, floor 2e-4 | `tools/oracle/z-image.py text` | text encoder |
| `z-image block` | block 0 matches the reference at its 19 internal boundaries: norms, projections, attention, SwiGLU | the rule, floor 2e-4 | `tools/oracle/z-image.py dit --size {size}` | DiT |
| `z-image dit` | one evaluation of the DiT, latent to velocity: embedders, refiners, 30 blocks, head, unpatchify | the rule, floor 5e-4; on a rectangle, the transposed golden must deviate; with --neural-engine, against bf16's own error + 25 % | `tools/oracle/z-image.py dit --size {size}; witness --size {size}; with --neural-engine, bf16 --size {size}` | DiT |
| `z-image trajectory` | the 8-step sampler: each step's velocity and latent, from the reference's own latent at that step | the rule on the guided pass, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/z-image.py trajectory --size {size}` | DiT |
| `z-image vae` | Flux's decoder on a rectangular latent (the centered crop of the final 1024² latent): what a square cannot see | PSNR ≥ 80 dB against the reference's image, and the transposed image must deviate | `tools/oracle/z-image.py vae --size {size}` | VAE |
| `qwen-image-2.1 text` | Qwen3-VL-8B reads the prompt (and with --edit N, the N pictures): template, identifiers, positions, 36 layers, prompt_embeds | identifiers and masks bit for bit; the rule, floor 5e-4 (with images, the deep layers judged on their text rows) | `tools/oracle/qwen-image-2.1.py text [--edit N]` | text encoder |
| `qwen-image-2.1 vision` | the processor (Lanczos, patches, grids), the 3D positions of both edit templates, and the vision tower with its deepstack, image by image | pixels, grids, identifiers and positions bit for bit; the rule, floor 5e-4 | `tools/oracle/qwen-image-2.1.py vision` | text encoder |
| `qwen-image-2.1 references` | a reference with transparency, prepared as the pipeline does: the VAE's RGBA input and the vision copy over white (at 512, Pillow premultiplies around its Lanczos; at 443 it copies) | bit for bit; the vision copy on its bytes (the Pil and Fast processors differ by 6e-8 in float) | `tools/oracle/qwen-image-2.1.py references [--side 443]` | — |
| `qwen-image-2.1 dit` | one evaluation three ways — prefill of the conditions, the step from their K/V cache, the full pass under the block-causal mask — plus the layout and the modulation | layout bit for bit; the rule, floor 5e-4, at the fp64 witness where one exists; cache and full pass ≤ 1e-5 apart; a second step the same bits; with --neural-engine, against bf16's own error + 25 %, cache and full pass too | `tools/oracle/qwen-image-2.1.py dit [--edit N]; witness; with --neural-engine, bf16` | DiT |
| `qwen-image-2.1 trajectory` | the 8 steps: σ, timesteps and noise to the bit, then each step's velocity from the oracle's latent and encoder states; the product's free run from the real inputs is shown | schedule and noise bit for bit; the rule on each guided velocity, floor 5e-4; the free run's final latent and image are shown, not judged | `tools/oracle/qwen-image-2.1.py trajectory [--edit N]` | DiT, text encoder, VAE |
| `qwen-image-2.1 vae` | the decoder on the trajectory's final latent (with --edit N, the edit's): unpacking and de-normalization to the bit, then the RGBA image plane by plane | vae_in bit for bit; the rule per plane, floor 5e-4, at the fp64 witness where one exists; on a rectangle, the transposed image must deviate | `tools/oracle/qwen-image-2.1.py vae [--edit N]; vae-witness` | VAE |
| `qwen-image-2.1 vae-encode` | the encoder on both edit pictures (RGBA): the raw mean and the normalized latent, plane by plane; the product's module (RGB in, alpha added) gives the same bits | the rule per plane, floor 5e-4, at the fp64 witness where one exists; the module bit for bit | `tools/oracle/qwen-image-2.1.py vae-encode; vae-witness` | VAE |
| `anima tokenizer` | the T5 tokenizer against `tokenizers` on a corpus of edge cases: normalization, whitespace, special tokens, truncation | every case bit for bit | `tools/oracle/anima.py tokenizer` | tokenizer |
| `anima text` | the prompt through both tokenizers, Qwen3-0.6B (final norm included) and the conditioner, to the DiT's [512, 1024] context | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/anima.py text` | text encoder, DiT (the conditioner) |
| `anima dit` | one evaluation of the DiT at σ = 1, boundary by boundary, and its output per latent plane against the fp64 witness | the rule, floor 5e-4; model_out per latent plane at the fp64 witness; on a rectangle, the transposed golden must deviate; with --neural-engine, against bf16's own error + 25 % | `tools/oracle/anima.py dit --size {size}; witness --size {size} --step 0; with --neural-engine, bf16 --size {size}` | DiT |
| `anima trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step | σ within 1e-6; the rule on each guided velocity, floor 5e-4; the free pass is shown with its gain, not judged | `tools/oracle/anima.py trajectory --size {size}` | DiT |
| `anima vae` | the Qwen-Image decoder on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/anima.py vae --size {size}; vae-witness --size {size}` | VAE |
| `krea2 text` | the prompt's identifiers, the 12 Qwen3-VL taps (padding removed, judged against states computed with it), then the fusion and txt_in on the oracle's taps | identifiers bit for bit; the rule, floor 5e-4 | `tools/oracle/krea2.py text` | text encoder, DiT (the fusion) |
| `krea2 dit` | one evaluation of the DiT at σ = 1, boundary by boundary, judged at the fp64 witness where it exists | the rule, floor 5e-4, at the fp64 witness where it exists, clouds when the oracle has draws; model_out per latent plane at the witness; on a rectangle, the transposed golden must deviate; with --neural-engine, against bf16's own error + 25 % | `tools/oracle/krea2.py dit --size {size}; witness --size {size}; witness-step --size {size} --step 0; draw --size {size} --step 0 --jitter 1 2 3; with --neural-engine, bf16 --size {size}` | DiT |
| `krea2 trajectory` | the schedule, then the 8 Euler steps: each step's velocity from the oracle's latent at that step, at the step's fp64 witness where it exists | σ within 1e-6; the rule on each guided velocity, floor 5e-4, at the step's fp64 witness where it exists, clouds when the oracle has draws; the free pass is shown, not judged | `tools/oracle/krea2.py trajectory --size {size}; witness-step --size {size} --step 0 1 2 3 4 5 6 7` | DiT |
| `krea2 vae` | the Qwen-Image decoder with Krea's weights on the final latent of the oracle's trajectory, plane by plane | the rule per plane, floor 5e-4, at the fp64 witness (no floor there); on a rectangle, the transposed image must deviate | `tools/oracle/krea2.py vae --size {size}; vae-witness --size {size}` | VAE |
