# CDIN_detection
# Exploiting Complementary Dynamic Incoherence for DeepFake Video Detection(CDIN)

A two-branch Complementary Dynamic Interaction Network (CDIN) is proposed to exploit both global (entire face) and local (mouth region) anomalous dynamic artifacts for DeepFake detection.

This is a PyTorch implementation of the paper.

## Installation
``` python
pip install -r requirements.txt
```
## Prepare Dataset
[FaceForensics++](https://github.com/ondyari/FaceForensics) as training dataset
``` python
python data_pre.py --dataset_path --output_path --subset --compression --imsize
```
## Training CDIN
``` python
python train.py
```

### Re-train FTCN
We also provide trainer to re-train FTCN model under ./FTCN/trainer
``` python
python FTCN/train_FTCN.py
```

