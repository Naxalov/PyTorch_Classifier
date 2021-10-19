import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

data_transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.ColorJitter(),
    transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor()
])
class CatDogDataset(Dataset):

  def __init__(self,path,transform = None):
    self.data_list = list(path.glob('*/*.jpg'))
    self.transform = transform
    
       
   
  def __getitem__(self,idx):
    img = Image.open(self.data_list[idx])
    if self.transform:
          img = self.transform(img)
    else:
        img = data_transform(img)
    # img = img.numpy().astype(np.float32)
    if 'Dog' in self.data_list[idx].parts:
      label = 1
    else:
      label = 0

    return img,label
  def __len__(self):
    return len(self.data_list)


def split_dataset(dataset,test_size):
  # Determine size of validation set
  dataset_n = len(dataset)
  test_n =int(dataset_n*test_size)
  train_n =dataset_n-test_n
  train_data ,test_data = random_split(dataset, [train_n, test_n], generator=torch.Generator().manual_seed(42))  
  
  #Pick first n_val indices for validation set
  return train_data,test_data

#Visualizing the Dataset
def show_data(data):
    # DATASET
    # Display image and label.
    img,label = data
    # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
    img = np.transpose(img, (1,2,0))
    if label==1:
        txt='Dog'
    else:
        txt='Cat'
    plt.title(txt)
    plt.imshow(img)
    plt.show()
    
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # # So we need to reshape it to (H, W, C):
    # img = train_features[0].permute(1, 2, 0)


    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    
def show_grid(dataset,save=False):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        if label==1:
            txt='Dog'
        else:
            txt='Cat'

        figure.add_subplot(rows, cols, i)
        plt.title(txt)
        plt.axis("off")
        # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
        img = np.transpose(img, (1,2,0))
        plt.imshow(img, cmap="gray")
    # Save as file
    if save:
        plt.savefig(fname='1.jpg',dpi=200)
    else:
        plt.show()