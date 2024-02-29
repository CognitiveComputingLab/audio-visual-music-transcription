import torch 
import torch.nn as nn 
import torch.nn.functional as F 

base_channel = 8
linear_nums = {
        'white':base_channel*2*15*2,
        'black':base_channel*2*10*2
}
class SimpleNet(nn.Module):
    def __init__(self,input_channel=1,num_classes=2,type='white'):
        super(SimpleNet,self).__init__()
        self.base = nn.Sequential(
                nn.Conv2d(input_channel,base_channel,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,stride=2),
                nn.Dropout(p=0.5),

                nn.Conv2d(base_channel,base_channel*2,kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,stride=2),
                nn.Dropout(p=0.5),
        )
        
        self.classifer = nn.Sequential(
                nn.Linear(linear_nums[type],256),
                nn.ReLU(inplace=True),
                #nn.Dropout(p=0.5),
                nn.Linear(256,num_classes)
       
        )
    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        x = self.classifer(x)
        return x 


if __name__=='__main__':
    inputs = torch.randn(1,1,60,10)
    net = SimpleNet()
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    out = net(inputs)
    print(out.size())
        
