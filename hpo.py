#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target,view_as(pred)).sum().item()
    print("\nAccuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    trained_iamges = 0
    num_images = len(train_loader.dataset)
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        trained_images += len(inputs)
        optimizer.step()
        print(f"{trained_images}/{num_images} images trained")
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_glad = False
        
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, 224))
        
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data = datasets.ImageFolder(data, transform=transform_functions)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",
                        type=int, 
                        default=64,
                        metavar="N",
                        help="input batch size for training (default: 64)",)
    parser.add_argument("--test-batch-size",
                       type=int,
                       default=1000,
                       metavar="N",
                       help="input batch size for testing (default: 1000)",)
    
    parser.add_argument("--epochs",
                       type=int,
                       default=14,
                       metavar="N",
                       help="number of epochs to train(default: 14)",)
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="Learning rate(default: 1.0)")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args = parser.parse_args()
    
    model=net()
    
    
    train_transforms = transforms.Compose([transforms.Resize((255, 255)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    
    test_transforms = transforms.Compose([transforms.Resize((255, 255)),
                                         transforms.ToTensor()])
    
    
    train_loader = create_data_loader(args.train, train_transforms, args.batch_size)
    test_loader = create_data_loader(args.test, test_transforms, args.test_batch_size, shuffle=False)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, loss_criterion, optimzer)
        test(model, test_loader)
        
    path = os.path.join(args.model_dir, "model.path")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    main()
