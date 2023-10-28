from time import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from copy import deepcopy

def train_test(model, trainload, testload, loss, optim, epochs, device):
    start_time = time()
    print(f'Start Time: {start_time}')

    for epoch in range(epochs):
        model.train()
        total_step = len(trainload)
        running_loss, running_corrects = 0, 0

        for i, (inputs, labels) in enumerate(trainload):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # train
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs) # prediction
                _loss = loss(outputs, labels) # loss
                _, pred = torch.max(outputs, 1)
                
                # backpropr
                _loss.backward()
                optim.step()

            running_loss += _loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred == labels.data)

            # display every 100 samples
            if (i * inputs.size(0)) % 1000 == 0 and i != 0:
                print(f'Epoch {epoch+1}/{epochs}, Step {(i * inputs.size(0))}/{len(trainload.dataset)}, '
                      f'Loss: {running_loss/(i * inputs.size(0)):.2f}, Acc: {running_corrects.double()/(i * inputs.size(0)):.2f}')



        # valdation
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = 0.0, 0.0

            for i, (inputs, labels) in enumerate(testload):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # test
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    _loss = loss(outputs, labels)

                valid_loss += _loss.item() * inputs.size(0)
                valid_acc += torch.sum(pred == labels.data)

            print(f'Epoch #{epoch+1}, '
                    f'Validation Loss: {valid_loss/len(testload.dataset):.2f}, '
                    f'Validation Acc: {valid_acc.double()/len(testload.dataset):.2f}')
            
    print(f'Training time: {start_time - time()}')
    print('Savin model')
    torch.save(model.state_dict(), 'my_shufflenet.pt')



def main():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # freeze all model params
    for params in model.parameters():
        params.requires_grad = False

    numftrs = model.fc.in_features
    model.fc = nn.Linear(numftrs, 10)

    model = model.to(device)

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261)
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261)
        )
    ])

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transforms)
    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)
    testloader = DataLoader(testset, batch_size=batch_size,shuffle=False)

    train_test(
        model=model,
        trainload=trainloader,
        testload=testloader,
        loss=nn.CrossEntropyLoss(),
        optim=optim.Adam(model.parameters(), lr=1e-3),
        epochs=35,
        device=device
    )
    print('==END==')

    
if __name__ == "__main__":

    main()