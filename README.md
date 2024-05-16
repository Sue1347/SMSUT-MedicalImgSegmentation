# Semi-supervised Multi-modal Medical Image Segmentation with Unified Translation 

To Run this file:

First Process the data into the nii files.

```bash
python data_pprocess/chaosPreparation.py
python data_pprocess/atlasPreparation.py
python data_pprocess/toPngAndSplit.py
```

Then, train and test the model
```bash
CUDA_VISIBLE_DEVICES=0 python trainer/uganConsisTrainer.py -p train -f 0
CUDA_VISIBLE_DEVICES=0 python trainer/uganConsisTrainer.py -p test -f 0 -i 000 -wh best
```

To cite this paper:

```
@article{SUN2024108570,
title = {Semi-supervised multi-modal medical image segmentation with unified translation},
journal = {Computers in Biology and Medicine},
volume = {176},
pages = {108570},
year = {2024},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.108570},
url = {https://www.sciencedirect.com/science/article/pii/S0010482524006541},
author = {Huajun Sun and Jia Wei and Wenguang Yuan and Rui Li}
}
```
