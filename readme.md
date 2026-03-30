# Weakly-Correlated Knowledge Distillation for Blind Image Quality Assessment [IEEE TMM 2025]
An Official Pytorch Implementation of Weakly-Correlated Knowledge Distillation for Blind Image Quality Assessment [IEEE TMM 2025].

## Get Started
### Datasets Preparation
- KonIQ-10K: Download the [KonIQ-10K](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.
- KADID-10K: Download the [KADID-10K](https://database.mmsp-kn.de/kadid-10k-database.html) dataset.
- LIVE: Download the [LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm) dataset.
- CSIQ: Download the [CSIQ](https://s2.smu.edu/~eclarson/csiq.html) dataset.
- TID2013: Download the [TID2013](https://www.ponomarenko.info/tid2013.htm) dataset.
- CLIVE: Download the [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/index.html) dataset.
- BID: The BID dataset may be difficult to find online, thanks to [@zwx8981](https://github.com/zwx8981) for providing the [BID](https://github.com/zwx8981/UNIQUE/tree/master?tab=readme-ov-file#link-to-download-the-bid-dataset) download link in their UNIQUE repository.

### Environment
```
#Prepare the environment via conda

torch 1.8+
torchvision
Python 3

# Install packages

pip install -r requirements.txt

```

## Models
We utilized the proposed weakly-correlated knowledge distillation method on six models. To demonstrate the effectiveness of the proposed method, we trained two versions: the Origin version and the ​WCKD version​. 

We also provide well-trained BIQA models and pre-trained segmentation model weights required by the WCKD method.

You can download them from this [link](https://pan.baidu.com/s/1hZjLdIDq4Ae_2qmIaY12IA?pwd=mg4y). 提取码: mg4y

## Folder Structures
Our repository includes six different projects corresponding to six baseline models. We outline the structure of each project and indicate the entry points for training and testing.
```
WCKD
└─ ResNet50
   └─ train_deeplab.py
   └─ test.py
└─ VGG
   └─ run.py
   └─ test.py
└─ VIT
   └─ run.py
   └─ test.py
└─ TReS
   └─ run.py
   └─ test.py
└─ HyperIQA
   └─ train_test_IQA.py
   └─ test.py
└─ LIQE
   └─ LIQE_Cov.py
   └─ test.py
```

## Train

### ResNet50 

-> Origin
```
python train_deeplab.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your save path' --cov False --fusion False
```

-> WCKD
```
python train_deeplab.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your save path' --cov True --fusion True
```

### TReS/HyperIQA/VIT/VGG

-> Origin
```
python run.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your save path' --cov False --fusion False
```

-> WCKD
```
python run.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your save path' --cov True --fusion True
```

### LIQE

-> Origin
```
python LIQE_Cov.py --sv_path 'Your save path' --cov False --fusion False
```

-> WCKD
```
python LIQE_Cov.py --sv_path 'Your save path' --cov True --fusion True
```


## Test

-> Origin
```
python test.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your checkpoint path' --cov False --fusion False
```

-> WCKD
```
python test.py --datapath '/IQA_Database/CSIQ/' --dataset 'CSIQ' --sv_path 'Your checkpoint path' --cov True --fusion True
```

## Ackknowledgement
This repo is developed based on TReS and LIQE. Please chheck [TReS](https://github.com/isalirezag/TReS) and [LIQE](https://github.com/zwx8981/LIQE?tab=readme-ov-file) for more details.

## Citation
If you find WCKD useful in your research, please consider citing:
```
@ARTICLE{11456770,
  author={Su, Ting and Huang, Lingze and Zou, Wenbin and Yue, Guanghui and Tian, Shishun},
  journal={IEEE Transactions on Multimedia}, 
  title={Weakly-Correlated Knowledge Distillation for Blind Image Quality Assessment}, 
  year={2026},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2026.3678061}}
```
