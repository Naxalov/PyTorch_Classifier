import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,random_split

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
  
  # Pick first n_val indices for validation set
  return train_data,test_data