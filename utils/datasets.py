from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image

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