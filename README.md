# FDL_AI-Generated_Images_Recognizer_Project
Project for FDL subject in PJAIT

# Dataset
[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

### Project structure
```
├── CIFAKE_ConvCNN_model.py
├── CIFAKE_ResNet-50_model.py
├── README.md
├── dataset
│   ├── test
│   │   ├── FAKE
│   │   └── REAL
│   └── train
│       ├── FAKE
│       └── REAL
└── training_results
    ├── conv_cnn
    │   ├── img.png
    │   └── conv_cnn.pth
    └── resnet-50
        ├── img.png
        └── resnet50.pth
```

## Research papers
1. [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.org/pdf/1912.11035) | [arxiv](https://arxiv.org/abs/1912.11035)
2. [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://www.researchgate.net/publication/377538637_CIFAKE_Image_Classification_and_Explainable_Identification_of_AI-Generated_Synthetic_Images)
3. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) 

## Model used for generating synthetic data
- [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Code | Documentations
1. [ResNet building](https://github.com/PeterWang512/CNNDetection/blob/master/networks/resnet.py#L183)
2. https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

# Training results
## Conv CNN [10 epochs]
- Time taken for training: 374.67s
- Precision: 0.9256
- Recall:	 0.9614
- F1 Score:	 0.9432

## ResNet 50 [5 epochs]
- Time taken for training: 2559.44s
- Precision: 0.9270
- Recall:	 0.9693
- F1 Score:	 0.9477

## Conv CNN with Tuner + DropOut + Weight Decay regularization [10 tries for 3 epochs + final 10 epochs training]
- Time taken for training: 1289.65s  
- Precision:	0.9513
- Recall:		0.9413
- F1 Score:	    0.9463

### BEST Configuration found with Tuner:
- batch_size: 32 
- n_neurons: 256
- dropout_p: 0.28
- lr: 0.001
- weight_decay: 0.0