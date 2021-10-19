import torchvision
import torch


def CustomVGG11(class_n=2):
    vgg = torchvision.models.vgg16(pretrained=False)
    vgg.classifier = torch.nn.Linear(25088,class_n)
    return vgg

