# Description
Cross-Domain Requirement Linking via Adversarial Domain Adaptation

- main.py contains the first phase (Requirements Linking Sample Construction) and the fourth phase (Target Linking Model Construction) of the RADIATION model
- core/pretrain.py contains the second phase of the model (Pre-training the Linking Model in the Source Domain) of the RADIATION model
- core/adapt.py contains the third stage of the model (Distance-enhanced Adversarial Representation Adaptation) of the RADIATION model
- The params folder contains model configuration information and running parameters
- data/processed folder contains the dataset mentioned in our paper, and we provide the processed version of the original dataset

# Packages
- torch
- pandas
- pytorch_pretrained_bert
- sklearn
- gensim
- torch

# Run the code
python main.py 

# Models
Models consisting of 1) an encoder trained on the source domain;  
2) a trained classifier on the source domain;  
3) a target domain encoder trained in the adversarial adaptation phase  
Models for migration from HIPPA Dataset to Easy Dataset for four components of RADIATION:  
link: https://pan.baidu.com/s/12ZAyIVu-qZvkAIDjAcwHtA; password: i7eq 



