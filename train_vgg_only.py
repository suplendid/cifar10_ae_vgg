import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import random
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import PIL.Image as Image
import os


from models import *
from utility import *

if __name__ == '__main__':
    
    transform_ae = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.05),
                transforms.ToTensor(),
                #transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 1/0.3), value='random'),
                ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    trainset = MyCifar10(path="./data", transform = transform_ae)
    val_rate = 0.2
    valid = True
    if valid:
        val_size = int(val_rate * len(trainset))
        train_size = len(trainset) - val_size
        print(train_size, val_size, len(trainset))
        train_data, val_data = random_split(trainset, [train_size, val_size])
        print(len(train_data), len(val_data))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, num_workers=2)#, sampler=weight_sampler(val_data))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=128,shuffle=True,num_workers=2)    
    
    for layer in range(1,6):
        net = VGG16_only(layer).cuda()
        print(net)
        total_epoch=200
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epoch)
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        for epoch in range(total_epoch):
            net.train()
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.cuda(), labels.cuda()
                optimizer.zero_grad()
                predicts = net(data)
                loss = criterion_cls(predicts, labels)
                loss.backward()
                optimizer.step()
                net.train_parameters['loss'] += loss.item()
                _, predicted = predicts.max(1)
                net.train_parameters['total'] += labels.size(0)
                net.train_parameters['correct'] += predicted.eq(labels).sum().item()
            net.train_parameters['loss'] = net.train_parameters['loss'] / (i+1)
            if net.train_parameters['loss'] < net.train_loss_min:
                net.train_loss_min = net.train_parameters['loss']
                _ = net.save_train()
            net.eval()
            with torch.no_grad():
                for i, (data, labels) in enumerate(val_loader):
                    data, labels = data.cuda(), labels.cuda()
                    predicts = net(data)
                    loss = criterion_cls(predicts, labels)
                    net.val_parameters['loss'] += loss.item()
                    _, predicted = predicts.max(1)
                    net.val_parameters['total'] += labels.size(0)
                    net.val_parameters['correct'] += predicted.eq(labels).sum().item()
            net.val_parameters['loss'] = net.val_parameters['loss'] / (i+1)
            print('[ epoch:%d ] train loss: %.5f val loss: %.5f train_acc:%.2f%% (%d/%d) val_acc:%.2f%% (%d/%d)' % (epoch + 1, net.train_parameters['loss'], net.val_parameters['loss'],
            100.*net.train_parameters['correct']/net.train_parameters['total'],net.train_parameters['correct'],net.train_parameters['total'], 100.*net.val_parameters['correct']/net.val_parameters['total'], net.val_parameters['correct'],net.val_parameters['total']))
            if net.val_parameters['loss'] < net.val_loss_min:
                net.val_loss_min = net.val_parameters['loss']
                net_path = net.save_val()
            train_loss_history.append(net.train_parameters['loss'])
            train_acc_history.append(100.*net.train_parameters['correct']/net.train_parameters['total'])
            val_loss_history.append(net.val_parameters['loss'])
            val_acc_history.append(100.*net.val_parameters['correct']/net.val_parameters['total'])
            net.reset_parameters()
            scheduler.step()
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                  "horse", "ship", "truck"]
        descript = torch.load(net_path)

        net.load_state_dict(descript['net'])
        class_correct = list(0 for i in range(10))
        class_total = list(0 for i in range(10))
        n = 0
        scm = np.zeros((10,10))

        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                images, labels = data[0].cuda(), data[1]
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                cm = confusion_matrix(labels, predicted, labels=list(np.arange(10)))
                scm += cm
                c = (predicted == labels).squeeze()
                n += len(c[c==True])
                for i in range(len(data[1])):
                    label = labels[i].item()
                    if c[i].item() == True:
                        class_correct[label] += 1
                    class_total[label] += 1

        x = 0
        for i in range(10):
            print('Accuracy of %5s : %2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
            x += class_correct[i] / class_total[i]
        print('Average Accuracy: %2f %%' % (n/100))

        plt.figure(figsize=(12,12))
        sns.heatmap(scm, annot=True, xticklabels = classes, yticklabels = classes, fmt=".1f", cmap="Blues")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'./images/heatmap_test_acc:{n/100}%_TotalLayer:{layer}_VGGonly.png')   
        
        plt.figure()
        plt.title("Loss History")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(train_loss_history,label="Train loss")
        plt.plot(val_loss_history,label="Val loss")
        plt.legend()
        plt.savefig(f'./images/loss_test_acc:{n/100}%_TotalLayer:{layer}_VGGonly.png')
        
        plt.figure()
        plt.title("Accuracy History")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(train_acc_history,label="Train acc")
        plt.plot(val_acc_history,label="Val acc")
        plt.legend()
        plt.savefig(f'./images/acc_test_acc:{n/100}%_TotalLayer:{layer}_VGGonly.png')
    