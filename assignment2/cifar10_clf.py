#author: Guan Song Wang
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets,transforms

if __name__=="__main__":

    learning_rate=0.001
    amsgrad=False
    epochs=10


    minst_tranform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.3081,))])

    train_minst=datasets.MNIST(root='./data/',train=True,download=True,
                               transform=minst_tranform)
    test_minst=datasets.MNIST(root='./data/',transform=minst_tranform,train=False)

    use_cuda=torch.cuda.is_available()
    torch.manual_seed(0)
    device=torch.device('cuda' if use_cuda else 'cpu')