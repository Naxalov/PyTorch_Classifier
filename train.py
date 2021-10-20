import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import dataset
from torchvision import transforms
import torchvision
from utils.datasets import CatDogDataset,split_dataset,show_data,show_grid
from pathlib import Path
from models.model import CustomVGG11,VGG11
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Doing computations on device = {}'.format(device))

# Data Preperation
DATA_DIR = Path('data/PetImages')

# dataset = CatDogDataset(DATA_DIR)

std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        std_normalize])
dataset = torchvision.datasets.ImageFolder('data/PetImages',transform=trans)


train_data,test_data = split_dataset(dataset=dataset,test_size=0.2)

# TO DO
# STATIC LABEL
print('number of dataset:',len(dataset))
print('Train:',len(train_data))

print('Val:',len(test_data))

# Visualizing the Dataset
# show_data(test_data[0])

#Iterate through the DataLoader

# show_grid(test_data,save=True)

# Preparing your data for training with DataLoader

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

# Visualizations single dataloader (Display image and label.)
# Iterate through the DataLoader
train_features, train_labels  = next(iter(train_dataloader))
show_data((train_features[0], train_labels[0]),save=True)


model = CustomVGG11().to(device)
# model = VGG11(num_classes=2).to(device)
x = torch.randn(2,3,224,224).to(device)
y = model(x)
print(y.size())


def validate(net, dataloader,loss_fn=nn.NLLLoss()):
    net.eval()
    count,acc,loss = 0,0,0
    with torch.no_grad():
        for features,labels in tqdm(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            lbls = labels #.to(default_device)
            out = net(features) #.to(default_device))
            loss += loss_fn(out,lbls) 
            pred = torch.max(out,1)[1]
            acc += (pred==lbls).sum()
            count += len(labels)
    return loss.item()/count, acc.item()/count


# loss and optimizer
learning_rate = 0.01
# Binary Cross Entropy between the target and the output:
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

num_epochs = 1
total_step = len(train_dataloader)

loss_list     = []
for epoch in range(num_epochs):
  for i,(train_x,train_y) in enumerate(tqdm(train_dataloader)):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    # forward pass
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss_list.append(loss)
    # backward
    loss.backward()
    # update
    optimizer.step()

    optimizer.zero_grad()
  vl,va = validate(model,test_dataloader,loss_fn=criterion)
  if i % 1 == 0:
    print(f'Epoch[{epoch+1}/{num_epochs}],\t Step [{i+1}/{total_step}] \t loss: {loss.item():.6f}, Val acc={va:.3f} Val loss={vl:.3f}')

def get_all_preds(model,loader):
  '''
  get pred and true label
  '''
  y_pred = []
  y_true = []
  with torch.no_grad():
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        y_pred.extend(preds.cpu().argmax(dim=1).numpy())
        y_true.extend(labels.cpu().numpy())  

  return y_true,y_pred

y_true, y_pred = get_all_preds(model, test_dataloader)
cm = confusion_matrix(y_true, y_pred)
print(cm)

# torch.save(model,'cats_dogs.pt')