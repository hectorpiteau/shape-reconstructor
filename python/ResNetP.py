class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0).to(device)
        self.batch_norm1 = nn.BatchNorm2d(out_channels).to(device)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1).to(device)
        self.batch_norm2 = nn.BatchNorm2d(out_channels).to(device)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0).to(device)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion).to(device)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU().to(device)
        
    def forward(self, input):
        identity = input.clone()
        
        input = self.relu(self.batch_norm1(self.conv1(input)))
        input = self.relu(self.batch_norm2(self.conv2(input)))
        
        input = self.batch_norm3(self.conv3(input))
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        
        # add identity
        input += identity
        input = self.relu(input)
        
        return input

class CustomLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(CustomLayer, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device)
        self.batch_norm1 = nn.BatchNorm2d(out_channels).to(device)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1).to(device)
        self.batch_norm2 = nn.BatchNorm2d(out_channels).to(device)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0).to(device)
        self.batch_norm3 = nn.BatchNorm2d(out_channels).to(device)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU().to(device)
        
    def forward(self, input):
        identity = input.clone()
        
        input = self.relu(self.batch_norm1(self.conv1(input)))
        input = self.relu(self.batch_norm2(self.conv2(input)))
        
        input = self.batch_norm3(self.conv3(input))
        
        #downsample if needed
        # if self.i_downsample is not None:
        #     identity = self.i_downsample(identity)
        
        # add identity
        input += identity
        input = self.relu(input)
        
        return input 

# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
class ResNetP(nn.Module):
    def __init__(self, num_classes=140, num_channels=3):
        super(ResNetP, self).__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_channels = 64
        self.num_classes = num_classes

        # Before first layer, pre process.
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=(2,2), padding=(3,3), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)


        # Define layers :
        self.lconv1 = nn.Conv2d(64, 1024, kernel_size=3, stride=1, padding=1).to(device)
        self.lbatch_norm1 = nn.BatchNorm2d(1024).to(device)
        
        self.lconv2 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1).to(device)
        self.lbatch_norm2 = nn.BatchNorm2d(2048).to(device)
        
        self.lconv3 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0).to(device)
        self.lbatch_norm3 = nn.BatchNorm2d(2048).to(device)
        
        self.lrelu = nn.ReLU().to(device)

        # self.layer1 = CustomLayer(64, 512)
        # self.layer2 = self._make_layer(Bottleneck, 4, planes=128, stride=2)
        # self.layer3 = self._make_layer(Bottleneck, 6, planes=256, stride=2)
        # self.layer4 = self._make_layer(Bottleneck, 3, planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Fully connected layer 
        self.fc = nn.Sequential(nn.Linear(2048, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 140)).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        
        x = self.lrelu(self.lbatch_norm1(self.lconv1(x)))
        x = self.lrelu(self.lbatch_norm2(self.lconv2(x)))
        x = self.lbatch_norm3(self.lconv3(x))
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            ).to(device)
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers).to(device)