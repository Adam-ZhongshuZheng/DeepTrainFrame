##########################################################################################
#   Model beta
#   The models for classify the breast img.
#
#   By Adam Zheng.
#   24 June, 2019
##########################################################################################


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


class ImageModel(nn.Module):

    def __init__(self, features, classifier='GAP', num_classes=2, init_weights=True):
        super(ImageModel, self).__init__()
        self.features = features
        self.cls = classifier
        if classifier == 'GAP':
            self.classifier = nn.Sequential(
                nn.Conv2d(512, num_classes, kernel_size=3, padding=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=4, stride=4)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if self.cls != 'GAP':
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M4':
            layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg_feature = {
    'A': [64, 'M4', 128, 'M4', 256, 512, 'M4'],
}


def image_model_fc(**kwargs):
    """
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 256, 256]             640
           BatchNorm2d-2         [-1, 64, 256, 256]             128
                  ReLU-3         [-1, 64, 256, 256]               0
             MaxPool2d-4           [-1, 64, 64, 64]               0
                Conv2d-5          [-1, 128, 64, 64]          73,856
           BatchNorm2d-6          [-1, 128, 64, 64]             256
                  ReLU-7          [-1, 128, 64, 64]               0
             MaxPool2d-8          [-1, 128, 16, 16]               0
                Conv2d-9          [-1, 256, 16, 16]         295,168
          BatchNorm2d-10          [-1, 256, 16, 16]             512
                 ReLU-11          [-1, 256, 16, 16]               0
               Conv2d-12          [-1, 512, 16, 16]       1,180,160
          BatchNorm2d-13          [-1, 512, 16, 16]           1,024
                 ReLU-14          [-1, 512, 16, 16]               0
            MaxPool2d-15            [-1, 512, 4, 4]               0
               Linear-16                 [-1, 4096]      33,558,528
                 ReLU-17                 [-1, 4096]               0
              Dropout-18                 [-1, 4096]               0
               Linear-19                    [-1, 2]           8,194
    ================================================================
    Total params: 35,118,466
    Trainable params: 35,118,466
    Non-trainable params: 0
    ----------------------------------------------------------------
    """
    model = ImageModel(make_layers(cfg_feature['A'], batch_norm=True), classifier='FC', **kwargs)
    return model


def image_model_gap(**kwargs):
    """
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 64, 256, 256]             640
           BatchNorm2d-2         [-1, 64, 256, 256]             128
                  ReLU-3         [-1, 64, 256, 256]               0
             MaxPool2d-4           [-1, 64, 64, 64]               0
                Conv2d-5          [-1, 128, 64, 64]          73,856
           BatchNorm2d-6          [-1, 128, 64, 64]             256
                  ReLU-7          [-1, 128, 64, 64]               0
             MaxPool2d-8          [-1, 128, 16, 16]               0
                Conv2d-9          [-1, 256, 16, 16]         295,168
          BatchNorm2d-10          [-1, 256, 16, 16]             512
                 ReLU-11          [-1, 256, 16, 16]               0
               Conv2d-12          [-1, 512, 16, 16]       1,180,160
          BatchNorm2d-13          [-1, 512, 16, 16]           1,024
                 ReLU-14          [-1, 512, 16, 16]               0
            MaxPool2d-15            [-1, 512, 4, 4]               0
               Conv2d-16              [-1, 2, 4, 4]           9,218
          BatchNorm2d-17              [-1, 2, 4, 4]               4
                 ReLU-18              [-1, 2, 4, 4]               0
            MaxPool2d-19              [-1, 2, 1, 1]               0
    ================================================================
    Total params: 1,560,966
    Trainable params: 1,560,966
    Non-trainable params: 0
    ----------------------------------------------------------------
    """
    model = ImageModel(make_layers(cfg_feature['A'], batch_norm=True), classifier='GAP', **kwargs)
    return model


if __name__ == '__main__':
    debug = True

    from torchsummary import summary

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = image_model_gap()
    net = image_model_fc()
    net = net.to(device)
    print(net)

    summary(net, (1, 256, 256))
