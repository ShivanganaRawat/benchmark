import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
    
    
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, norm_layer=nn.BatchNorm2d, drop_conv=0.0):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)

        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False), 
                norm_layer(planes * self.expansion),
                nn.Dropout2d(drop_conv)
            )

        self.drop = nn.Sequential()
        if drop_conv > 0.0:
            self.drop = nn.Dropout2d(drop_conv)

    def forward(self, x):
        out = self.drop(self.bn1(self.conv1(x))).relu_()
        out = self.drop(self.bn2(self.conv2(out))).relu_()
        out = self.drop(self.bn3(self.conv3(out)))
          
        x = self.downsample(x)
        return out.add_(x).relu_()


class ResNet(nn.Module):

    def __init__(self, layers, num_classes=1000, groups=1, width_per_group=64, 
                 drop_conv=0.0, drop_fc=0.0, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        block = Bottleneck

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(drop_conv)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_conv=drop_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_conv=drop_conv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_conv=drop_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_conv=drop_conv)
        self.drop = nn.Dropout(drop_fc)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                if isinstance(m.bn3, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride=1, drop_conv=0.0):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride=stride, groups=self.groups,
                base_width=self.base_width, norm_layer=self._norm_layer, drop_conv=drop_conv))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean((2,3))
        x = self.drop(x)
        x = self.fc(x)
        return x

def resnet50(num_classes=1000, drop_conv=0.0, drop_fc=0.0, norm_layer=nn.BatchNorm2d):
    return ResNet([3, 4, 6, 3], num_classes=num_classes, norm_layer=norm_layer,
            drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def resnet101(num_classes=1000, drop_conv=0.1, drop_fc=0.1, norm_layer=nn.BatchNorm2d):
    model =  ResNet([3, 4, 23, 3], num_classes=num_classes, norm_layer=norm_layer,
            drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)
    state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
    model.load_state_dict(state_dict)
    return model

def resnet200(num_classes=1000, drop_conv=0.0, drop_fc=0.0, norm_layer=nn.BatchNorm2d):
    return ResNet([3,24,36,3], num_classes=num_classes, norm_layer=norm_layer,
            drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def resnext50_32x4d(num_classes=1000, drop_conv=0.0, drop_fc=0.0, norm_layer=nn.BatchNorm2d):
    return ResNet([3, 4, 6, 3], num_classes=num_classes, norm_layer=norm_layer,
            drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=4)

def resnext101_32x8d(num_classes=1000, drop_conv=0.0, drop_fc=0.0, norm_layer=nn.BatchNorm2d):
    return ResNet([3, 4, 23, 3], num_classes=num_classes, norm_layer=norm_layer,
            drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=8)
