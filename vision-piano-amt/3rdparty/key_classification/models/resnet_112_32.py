import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

class ResidualBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride = 1,need_shortcut = False):
        super(ResidualBlock,self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size = 3,stride = stride,padding = 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True),
            nn.Conv2d(outchannels,outchannels,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(outchannels)
         )
        if need_shortcut:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannels,outchannels,kernel_size = 1,stride = stride),
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.short_cut = nn.Sequential()
    
    def forward(self,x):
        out = self.right(x)
        out += self.short_cut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self,input_channel=1, base_channel=8,num_classes=2):
        super(ResNet18,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(input_channel, base_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.make_layers = nn.Sequential(
            ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=True),
            ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=False),
            ResidualBlock(base_channel,base_channel*2,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*2,base_channel*2,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*2,base_channel*4,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*4,base_channel*4,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*4,base_channel*8,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*8,base_channel*8,stride=1,need_shortcut=False)
        )
        self.fc = nn.Linear(base_channel*8*4,num_classes)
        self.num_classes = num_classes

    def forward(self,x):
        out = self.pre_layer(x)
        out = self.make_layers(out)
        out = F.dropout(out,p=0.5)
        out = out.view(-1,self.num_flatters(out))
        return self.fc(out)

    def num_flatters(self,x):
        sizes = x.size()[1:]
        result = 1
        for size in sizes:
            result *= size
        return result 


if __name__=='__main__':
    input = torch.randn(12,1,112,32)
    net = ResNet18()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    stime = time.time()
    output = net(input)
    print(output.size())
    print('cost {}'.format(time.time()-stime))
