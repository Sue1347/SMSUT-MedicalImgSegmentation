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
