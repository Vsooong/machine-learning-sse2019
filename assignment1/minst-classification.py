import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets,transforms
class minstNet(nn.Module):
    def __init__(self):
        super(minstNet, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1, 32,kernel_size=5,stride=1,padding=0),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv2=nn.Sequential(nn.Conv2d(32,64,3,1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2,stride=2))

        self.dense=nn.Sequential(nn.Linear(5*5*64,800),
                                 nn.ReLU(),
                                 nn.Linear(800,10))


    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(-1,5*5*64)
        x=self.dense(x)
        x=F.log_softmax(x,dim=1)

        return x


def check_dataset():
    import matplotlib.pyplot as plt

    print(len(train_minst))
    images,labels=next(iter(train_loader))
    images=torchvision.utils.make_grid(images,8,8)
    images=images.numpy().transpose(1,2,0)
    std=[0.1307]
    mean=[0.3081]
    imgs=images*std+mean
    print(labels)
    plt.imshow(imgs)
    plt.show()

def train(model,device,train_loader,optimizer,epoch,loss_function):
    model.train()
    for batch_index,data in enumerate(train_loader):
        data,target=data
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output=model(data)

        loss=loss_function(output,target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":

    learning_rate=0.001
    amsgrad=False
    epochs=10


    minst_tranform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])

    train_minst=datasets.MNIST(root='./data/',train=True,download=True,
                               transform=minst_tranform)
    test_minst=datasets.MNIST(root='./data/',transform=minst_tranform,train=False)

    use_cuda=torch.cuda.is_available()
    torch.manual_seed(0)
    device=torch.device('cuda' if use_cuda else 'cpu')

    train_loader=torch.utils.data.DataLoader(train_minst,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_minst,batch_size=256,shuffle=False)

    model=minstNet().to(device)
    loss_function = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate,amsgrad=amsgrad)


    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer=optimizer, epoch=epoch, loss_function=loss_function)
        test(model,device,test_loader)
    torch.save(model.state_dict(),'./minst_clf.pt')

    # check_dataset()
    #
    # x=train_minst[0][0]
    # x=torch.from_numpy(np.expand_dims(x,axis=0))
    # x=x.to(device)
    # model.forward(x)






