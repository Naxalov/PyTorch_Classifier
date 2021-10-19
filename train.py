from torch.utils.data import dataset
from utils.datasets import CatDogDataset,split_dataset
from pathlib import Path

# Data Preperation
DATA_DIR = Path('data/PetImages')

dataset = CatDogDataset(DATA_DIR)

train_data,test_data = split_dataset(dataset=dataset,test_size=0.2)

# TO DO
# STATIC LABEL
print('number of dataset:',len(dataset))
print('Train:',len(train_data))

print('Val:',len(test_data))