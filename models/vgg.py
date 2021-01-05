import torch
import torch.nn as nn
import numpy as np
import os

vgg16_layers = {
                0:[64, 64, 'M'],
                1:[128, 128, 'M'],
                2:[256, 256, 256, 'M'],
                3:[512, 512, 512, 'M'],
                4:[512, 512, 512, 'M']
               }

class VGG16(nn.Module):
    def __init__(self, in_channels, total_layers):
        super(VGG16, self).__init__()
        self.features = self.make_feature_layers(in_channels, total_layers, vgg16_layers)
        self.classifier = nn.Linear(vgg16_layers[total_layers-1][0], 10)
    def make_feature_layers(self, in_channels, total_layers, vgg_layers):
        layers = []
        for i in range(total_layers):
            for l in vgg_layers[i]:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                else:
                    layers += [
                        nn.Conv2d(in_channels, l, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(l),
                        nn.ReLU(inplace=True)
                    ]
                    in_channels = l
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#1   64*16*16 = 16384
#2   128*8*8 = 8192
#3   256*4*4 = 4096
#4   512*2*2 = 2048
#5   512*1*1 = 512

class VGG16_only(nn.Module):
    def __init__(self, total_layers):
        super(VGG16_only, self).__init__()
        self.features = self.make_feature_layers(total_layers, vgg16_layers)
        self.classifier = self.make_fc(total_layers)
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.train_loss_min = np.Inf
        self.val_loss_min = np.Inf
        self.total_layers = total_layers
    def make_fc(self, total_layers):
        layers = []
        if total_layers == 1:
            layers += [
                nn.Linear(16384, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 2:
            layers += [
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 3:
            layers += [
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 4:
            layers += [nn.Linear(2048,10)]
        elif total_layers == 5:
            layers += [nn.Linear(512,10)]
        return nn.Sequential(*layers)
    def make_feature_layers(self, total_layers, vgg_layers):
        layers = []
        in_channels = 3
        for i in range(total_layers):
            for l in vgg_layers[i]:
                if l == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
                else:
                    layers += [
                        nn.Conv2d(in_channels, l, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(l),
                        nn.ReLU(inplace=True)
                    ]
                    in_channels = l
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    
    def save_train(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.train_parameters['correct'] / self.train_parameters['total'] *100}%"
        save_dict['total_layers'] = f'{self.total_layers}'
        
        save_path = f'check_points/train:layers:{self.total_layers}_VGGonly.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def save_val(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.val_parameters['correct'] / self.val_parameters['total'] *100}%"
        save_dict['total_layers'] = f'{self.total_layers}'
        
        save_path = f'check_points/val:layers:{self.total_layers}_VGGonly.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def reset_parameters(self):
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}

class VGG16_classifier(nn.Module):
    def __init__(self, total_layers):
        super(VGG16_classifier, self).__init__()
        self.classifier = self.make_fc(total_layers)
        self.total_layers = total_layers
    def make_fc(self, total_layers):
        layers = []
        if total_layers == 1:
            layers += [
                nn.Linear(16384, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 2:
            layers += [
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 3:
            layers += [
                nn.Linear(4096,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,10),
            ]
        elif total_layers == 4:
            layers += [nn.Linear(2048,10)]
        elif total_layers == 5:
            layers += [nn.Linear(512,10)]
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    
    
def test():
    vgg16 = VGG16_classifier(total_layers = 3)
    x = torch.randn(128,256,4,4)
    y = vgg16(x)
    print(y.size())
#test()