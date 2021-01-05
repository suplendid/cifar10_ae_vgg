from models import *
import torch
import torch.nn as nn

class AE_VGG(nn.Module):
    def __init__(self, autoencoder_layers):
        super(AE_VGG, self).__init__()
        self.autoencoder = AutoEncoder(autoencoder_layers)
        self.vgg16 = VGG16(int(64*(2**(autoencoder_layers-1))), 5-autoencoder_layers)
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.train_loss_min = np.Inf
        self.val_loss_min = np.Inf
        self.ae_layers = autoencoder_layers
    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        classification = self.vgg16(encoded)
        
        return encoded, decoded, classification
    def save_train(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.train_parameters['correct'] / self.train_parameters['total'] *100}%"
        save_dict['ae_layers'] = f'{self.ae_layers}'
        save_dict['vgg_layers'] = f'{5-self.ae_layers}'
        
        save_path = f'check_points/train:ae_layers:{self.ae_layers}_AE_VGG_end2end.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def save_val(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.val_parameters['correct'] / self.val_parameters['total'] *100}%"
        save_dict['ae_layers'] = f'{self.ae_layers}'
        save_dict['vgg_layers'] = f'{5-self.ae_layers}'
        
        save_path = f'check_points/val:ae_layers:{self.ae_layers}_AE_VGG_end2end.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def reset_parameters(self):
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}

class AEVGG_classifier(nn.Module):
    def __init__(self, autoencoder_layers):
        super(AEVGG_classifier, self).__init__()
        self.autoencoder = AutoEncoder_VGG(autoencoder_layers)
        self.vgg16 = VGG16_classifier(autoencoder_layers)
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.train_loss_min = np.Inf
        self.val_loss_min = np.Inf
        self.ae_layers = autoencoder_layers
    def forward(self, x):
        encoded, decoded = self.autoencoder(x)
        classification = self.vgg16(encoded)
        
        return encoded, decoded, classification
    def save_train(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.train_parameters['correct'] / self.train_parameters['total'] *100}%"
        save_dict['ae_layers'] = f'{self.ae_layers}'
        save_dict['vgg_layers'] = f'{5-self.ae_layers}'
        
        save_path = f'check_points/train:ae_layers:{self.ae_layers}_AE_VGG_end2end_2.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def save_val(self):
        save_dict = dict()
        save_dict['net'] = self.state_dict()
        save_dict['acc'] = f"{self.val_parameters['correct'] / self.val_parameters['total'] *100}%"
        save_dict['ae_layers'] = f'{self.ae_layers}'
        save_dict['vgg_layers'] = f'{5-self.ae_layers}'
        
        save_path = f'check_points/val:ae_layers:{self.ae_layers}_AE_VGG_end2end_2.pt'
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(save_dict, save_path)
        return save_path
    def reset_parameters(self):
        self.train_parameters = {'loss':0.0, 'correct':0, 'total':0}
        self.val_parameters = {'loss':0.0, 'correct':0, 'total':0}
