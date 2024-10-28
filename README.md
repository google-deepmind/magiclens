# MagicLens

This repo contains implementation of MagicLens. The code here uses Jax and Flax.
Note that the current implementation does not yet support training.
Refer to the [website](https://open-vision-language.github.io/MagicLens/) for dataset examples.

## Abstract

We introduce MagicLens, a series of self-supervised image retrieval models that support
open-ended instructions. The core thesis of MagicLens is that text
instructions can enable retrieving images with
richer relations beyond visual similarity. MagicLens is built on a
key novel insight: image pairs that naturally occur
on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we
can bring those implicit relations explicit by synthesizing instructions via large multimodal models (LMMs) and large language models (LLMs).
Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations
mined from the web, MagicLens achieves comparable or better results on eight benchmarks of
various image retrieval tasks than prior state-of-the-art (SOTA) methods. Remarkably, it outperforms previous SOTA but with a 50× smaller
model size on multiple benchmarks. Additional
human analyses on a 1.4M-image unseen corpus
further demonstrate the diversity of search intents
supported by MagicLens.
![Intro image](https://open-vision-language.github.io/MagicLens/static/images/magiclens_overview.png)

## Setup
```
conda create --name magic_lens python=3.9
conda activate magic_lens
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
# you may need to install corresponding GPU version of jax following https://jax.readthedocs.io/en/latest/installation.html
# e.g.,
# # CUDA 12 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # CUDA 11 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Model Download
Download model via:
```
cd .. # in main folder `magiclens`
# you may need to use `gcloud auth login` for access, any gmail account should work.
gsutil cp -R gs://gresearch/magiclens/models ./
```

OR via [google drive](https://drive.google.com/drive/folders/1MXszMqIIh-yV7cYxWUxP7uHs9gfuTT3u)

### Data Preparation
Please follow each dataset folder in `./data`. Currently we have successfully tested FIQ and CIRCO:

## Inference
```
python inference.py \
--model_size large \
--model_path ./models/magic_lens_clip_large.pkl \
--dataset circo

```

Due to the weight conversion, the performance may be slightly different:

In `CIRCO`
| Model | map@5 | map@10 | map@25 | map@50 |
|----------|----------|----------|----------|----------|
| Prior SOTA | 26.8 | 27.6 | 30.0 | 31.0 |
| Base (original) | 23.1 | 23.8 | 25.8 | 26.7 |
| Base (converted) | 22.3 | 23.2 | 25.0 | 26.0 |
| Large (original) | 29.6 | 30.8 | 33.4 | 34.4 |
| Large (converted) | 29.5 | 30.8 | 33.2 | 34.3 |

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@inproceedings{zhang2024magiclens,
  title = 	 {{M}agic{L}ens: Self-Supervised Image Retrieval with Open-Ended Instructions},
  author =       {Zhang, Kai and Luan, Yi and Hu, Hexiang and Lee, Kenton and Qiao, Siyuan and Chen, Wenhu and Su, Yu and Chang, Ming-Wei},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {59403--59420},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/zhang24an.html}
}

```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
