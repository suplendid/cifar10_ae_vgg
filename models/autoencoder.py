import torch
import torch.nn as nn

vgg16_layers = {
                0:[64, 64, 'M'],
                1:[128, 128, 'M'],
                2:[256, 256, 256, 'M'],
                3:[512, 512, 512, 'M'],
                4:[512, 512, 512, 'M']
               }

class AutoEncoder(nn.Module):
    def __init__(self, total_layers):
        super(AutoEncoder, self).__init__()
        self.encoder = self.make_encoder(total_layers)
        self.decoder = self.make_decoder(total_layers)
    def make_encoder(self, total_layers):
        layers = []
        in_channel = 3
        out_channel = 64
        for i in range(total_layers):
            layers += [
                nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ]
            in_channel = out_channel
            out_channel = int(out_channel *2)
        return nn.Sequential(*layers)
    def make_decoder(self, total_layers):
        layers = []
        in_channel = int(64*(2**(total_layers-1)))
        out_channel = in_channel//2
        for i in range(total_layers):
            if i == (total_layers -1):
                out_channel = 3
            layers += [
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel//2
        return nn.Sequential(*layers)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoder_VGG(nn.Module):
    def __init__(self, total_layers):
        super(AutoEncoder_VGG, self).__init__()
        self.encoder = self.make_encoder(total_layers, vgg16_layers)
        self.decoder = self.make_decoder(total_layers)
    def make_encoder(self, total_layers, vgg_layers):
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
    def make_decoder(self, total_layers):
        layers = []
        in_channel = int(64*(2**(total_layers-1)))
        out_channel = in_channel//2
        for i in range(total_layers):
            if i == (total_layers -1):
                out_channel = 3
            layers += [
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel//2
        return nn.Sequential(*layers)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
def test():
    autoencoder = AutoEncoder_VGG(3)
    x = torch.randn(128,3,32,32)
    encoded, decoded = autoencoder(x)
    print(encoded.size(), decoded.size())
#test()