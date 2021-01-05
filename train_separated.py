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
from end_2_end import AE_VGG

if __name__ == '__main__':
    
    def train():
        global train_loss_min
        vgg.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zero the parameter gradients
            #inputs = autoencoder(Variable(inputs))[0]
            inputs = autoencoder(inputs.cuda())[0]
            optimizer_cls.zero_grad()
            outputs = vgg(inputs)
            loss = criterion_cls(outputs, labels)
            loss.backward()
            optimizer_cls.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        train_loss = train_loss / (i+1)
        if train_loss < train_loss_min:
            state = {
                    'net': vgg.state_dict(),
                    'acc': f'{100.*train_correct/train_total}%',
            }
            torch.save(state, vgg_path_train)
            train_loss_min = train_loss
        return train_loss, train_correct, train_total
    def validate(train_loss, train_correct, train_total):
        global valid_loss_min
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        vgg.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].cuda(), data[1].cuda()
                #inputs = autoencoder(Variable(inputs))[0]
                inputs = autoencoder(inputs.cuda())[0]
                outputs = vgg(inputs)
                loss = criterion_cls(outputs, labels)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_loss += loss.item()
        val_loss = val_loss / (i+1)
        print('[ epoch:%d ] train loss: %.5f val loss: %.5f train_acc:%.2f%% (%d/%d) val_acc:%.2f%% (%d/%d)' % (epoch + 1, train_loss, val_loss,
                100.*train_correct/train_total,train_correct,train_total, 100.*val_correct/val_total, val_correct,val_total))
        if val_loss < valid_loss_min:
            state = {
                'net': vgg.state_dict(),
                'acc': f'{100.*val_correct/val_total}%',
            }
            torch.save(state, vgg_path_val)
            valid_loss_min = val_loss
        return val_loss, 100.*val_correct/val_total
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
    
    for auto_layer in range(1,5):
        vgg_path_train = f'check_points/vgg_train_{5-auto_layer}.pt'
        vgg_path_val = f'check_points/vgg_val_{5-auto_layer}.pt'
        autoencoder = AutoEncoder(auto_layer).cuda()
        vgg = VGG16(int(64*(2**(auto_layer-1))), 5-auto_layer).cuda()
        total_epoch = 200
        criterion_ae = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        optimizer_ae = optim.SGD(autoencoder.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        optimizer_cls = optim.SGD(vgg.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler_ae = CosineAnnealingLR(optimizer_ae, T_max=total_epoch)
        scheduler_cls = CosineAnnealingLR(optimizer_cls, T_max=total_epoch)
        loss_min = np.Inf
        ae_path = f'check_points/ae_{auto_layer}.pt'
        for epoch in range(total_epoch):
            train_loss = 0.0
            test_loss = 0.0
            autoencoder.train()
            for i, (inputs, _) in enumerate(train_loader, 0):
                #inputs = get_torch_vars(inputs)
                inputs = inputs.cuda()
                optimizer_ae.zero_grad()
                # ============ Forward ============
                encoded, outputs = autoencoder(inputs)
                loss = criterion_ae(outputs, inputs)
                # ============ Backward ============
                loss.backward()
                optimizer_ae.step()
                train_loss += loss.item()
            train_loss = train_loss / (i+1)
            autoencoder.eval()
            with torch.no_grad():
                for i, (inputs, _) in enumerate(val_loader, 0):
                    #inputs = get_torch_vars(inputs)
                    inputs = inputs.cuda()
                    # ============ Forward ============
                    encoded, outputs = autoencoder(inputs)
                    loss = criterion_ae(outputs, inputs)

                    test_loss += loss.item()
            test_loss = test_loss / (i+1)
            print(f"[epoch:{epoch}] train_loss:{train_loss:.9f} test_loss:{test_loss:.9f}")
            if loss_min > test_loss:
                torch.save(autoencoder.state_dict(), ae_path)
                loss_min = test_loss
            scheduler_ae.step()
        print("finish training ae")
        autoencoder.load_state_dict(torch.load(ae_path))
        valid_loss_min = np.Inf
        train_loss_min = np.Inf
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        for epoch in range(total_epoch):
            train_loss, train_correct, train_total = train()
            train_loss_history.append(train_loss)
            train_acc_history.append(100.*train_correct/train_total)
            val_loss, val_acc = validate(train_loss, train_correct, train_total)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            scheduler_cls.step()
        print('Finished Training') 
        
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                  "horse", "ship", "truck"]
        descript = torch.load(vgg_path_val)

        vgg.load_state_dict(descript['net'])
        class_correct = list(0 for i in range(10))
        class_total = list(0 for i in range(10))
        n = 0
        scm = np.zeros((10,10))

        vgg.eval()
        test_loss_cls = 0.0
        test_loss_ae = 0.0
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                images, labels = data[0].cuda(), data[1].cuda()
                encoded, decoded = autoencoder(images)
                outputs = vgg(encoded)
                loss_ae = criterion_ae(images, decoded)
                loss_cls = criterion_cls(outputs, labels)
                test_loss_ae += loss_ae.item()
                test_loss_cls += loss_cls.item()
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                labels = labels.cpu()
                cm = confusion_matrix(labels, predicted, labels=list(np.arange(10)))
                scm += cm
                c = (predicted == labels).squeeze()
                n += len(c[c==True])
                for i in range(len(data[1])):
                    label = labels[i].item()
                    if c[i].item() == True:
                        class_correct[label] += 1
                    class_total[label] += 1
        test_loss_ae = test_loss_ae / (j+1)
        test_loss_cls = test_loss_cls / (j+1)
        
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
        plt.savefig(f'./images/heatmap_test_acc:{n/100}%_AELayer:{auto_layer}_AELoss:{test_loss_ae:.3f}_VGGLoss:{test_loss_cls:.3f}_AE_VGG_separated.png')   
        
        plt.figure()
        plt.title("Loss History")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(train_loss_history,label="Train loss")
        plt.plot(val_loss_history,label="Val loss")
        plt.legend()
        plt.savefig(f'./images/loss_test_acc:{n/100}%_AELayer:{auto_layer}_AELoss:{test_loss_ae:.3f}_VGGLoss:{test_loss_cls:.3f}_AE_VGG_separated.png')
        
        plt.figure()
        plt.title("Accuracy History")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(train_acc_history,label="Train acc")
        plt.plot(val_acc_history,label="Val acc")
        plt.legend()
        plt.savefig(f'./images/acc_test_acc:{n/100}%_AELayer:{auto_layer}_AELoss:{test_loss_ae:.3f}_VGGLoss:{test_loss_cls:.3f}_AE_VGG_separated.png')
        