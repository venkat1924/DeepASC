import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import numParams


class GLSTM(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=1024, groups=2):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups
     
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
     
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
        self.mid_features = mid_features
     
    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)
    
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()
      
        return out
     

class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=(0,0)):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out
       
            
class Net(nn.Module):
    def __init__(self, scale=2, lstm=True, in_channels=50):
        super(Net, self).__init__()

        # self.conv0 = None
        # if in_channels != 2:
        #     self.conv0 = GluConv2d(in_channels=in_channels, out_channels=2, kernel_size=(1,3), stride=(1,2))
        self.is_medium = scale != 2
        scale = 2
        self.conv1 = GluConv2d(in_channels=in_channels, out_channels=in_channels*scale, kernel_size=(1,2), stride=(1,2))
        self.conv2 = GluConv2d(in_channels=in_channels*scale, out_channels=in_channels*(scale**2), kernel_size=(1,2), stride=(1,2))
        self.conv3 = GluConv2d(in_channels=in_channels*(scale**2), out_channels=in_channels*(scale**3), kernel_size=(1,2), stride=(1,2))
        self.conv4 = GluConv2d(in_channels=in_channels*(scale**3), out_channels=in_channels*(scale**4), kernel_size=(1,2), stride=(1,2))

        self.conv5_t_1 = GluConvTranspose2d(in_channels=in_channels*scale**4, out_channels=in_channels*scale**3, kernel_size=(1,2), stride=(1,2))
        self.conv4_t_1 = GluConvTranspose2d(in_channels=in_channels*scale**3, out_channels=in_channels*scale**2, kernel_size=(1,2), stride=(1,2))
        self.conv3_t_1 = GluConvTranspose2d(in_channels=in_channels*scale**2, out_channels=in_channels*scale, kernel_size=(1,2), stride=(1,2))
        self.conv2_t_1 = GluConvTranspose2d(in_channels=in_channels*scale, out_channels=in_channels, kernel_size=(1,2), stride=(1,2))
        # self.conv2_t_1 = GluConvTranspose2d(in_channels=scale*4, out_channels=scale, kernel_size=(1,3), stride=(1,2), output_padding=(0,1))
        # self.conv1_t_1 = GluConvTranspose2d(in_channels=scale*2, out_channels=1, kernel_size=(1,3), stride=(1,2))
        
        # self.conv5_t_2 = GluConvTranspose2d(in_channels=in_channels, out_channels=scale*8, kernel_size=(1,3), stride=(1,2))
        # self.conv4_t_2 = GluConvTranspose2d(in_channels=scale*16, out_channels=scale*4, kernel_size=(1,3), stride=(1,2))
        # self.conv3_t_2 = GluConvTranspose2d(in_channels=scale*8, out_channels=scale*2, kernel_size=(1,3), stride=(1,2))
        # self.conv2_t_2 = GluConvTranspose2d(in_channels=scale*4, out_channels=scale, kernel_size=(1,3), stride=(1,2), output_padding=(0,1))
        # self.conv1_t_2 = GluConvTranspose2d(in_channels=scale*2, out_channels=1, kernel_size=(1,3), stride=(1,2))
        
        # self.in_bn = nn.BatchNorm2d(2)

        self.bn1 = nn.BatchNorm2d(in_channels*scale)
        self.bn2 = nn.BatchNorm2d(in_channels*(scale**2))
        self.bn3 = nn.BatchNorm2d(in_channels*(scale**3))
        self.bn4 = nn.BatchNorm2d(in_channels*(scale**4))

        self.bn5_t_1 = nn.BatchNorm2d(in_channels*(scale**3)) 
        self.bn4_t_1 = nn.BatchNorm2d(in_channels*(scale**2))
        self.bn3_t_1 = nn.BatchNorm2d(in_channels*scale)
        self.bn2_t_1 = nn.BatchNorm2d(in_channels)

        self.conv5 = None
        self.conv6_t_1 = None
        self.bn5 = None
        self.bn6_t_1 = None
        if self.is_medium:
            self.conv5 = GluConv2d(in_channels=in_channels*(scale**4), out_channels=in_channels*(scale**5), kernel_size=(1,2), stride=(1,2))
            self.conv6_t_1 = GluConvTranspose2d(in_channels=in_channels*(scale**5), out_channels=in_channels*(scale**4), kernel_size=(1,2), stride=(1,2))
            self.bn5 = nn.BatchNorm2d(in_channels*(scale**5))
            self.bn6_t_1 = nn.BatchNorm2d(in_channels*(scale**4))
        # self.bn2_t_1 = nn.BatchNorm2d(scale)
        # self.bn1_t_1 = nn.BatchNorm2d(1)

        # self.bn5_t_2 = nn.BatchNorm2d(scale*8)
        # self.bn4_t_2 = nn.BatchNorm2d(scale*4)
        # self.bn3_t_2 = nn.BatchNorm2d(scale*2)
        # self.bn2_t_2 = nn.BatchNorm2d(scale)
        # self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        
        # self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels)
        # self.fc2 = nn.Linear(in_features=161, out_features=161)

    def forward(self, x):
        batches = x.shape[0]//50
        x = x.reshape(batches, 50, x.shape[1], x.shape[2])
        e1 = self.elu(self.bn1(self.conv1(x)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        out = self.elu(self.bn3(self.conv3(e2)))
        out = self.elu(self.bn4(self.conv4(out)))

        if self.is_medium:
            out = self.elu(self.bn5(self.conv5(out)))
            out = self.elu(self.bn6_t_1(self.conv6_t_1(out)))

        out = self.elu(self.bn5_t_1(self.conv5_t_1(out)))
        d4_1 = self.elu(self.bn4_t_1(self.conv4_t_1(out)))
        d3_1 = self.elu(self.bn3_t_1(self.conv3_t_1(d4_1)))
        out1 = self.elu(self.bn2_t_1(self.conv2_t_1(d3_1)))
        # d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))
        
        # d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e4), dim=1))
        # d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e3), dim=1))
        # d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e2), dim=1))
        # d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e1), dim=1))
        # d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))
        
        # out1 = self.fc1(out1)
        # out2 = self.fc2(d1_2)
        # out = torch.cat([out1, out2], dim=1)
        out = out1.reshape(out1.shape[0]*out1.shape[1], out1.shape[2], out1.shape[3])
        return out1

 
def test_model():
    net = Net()
    param_count = numParams(net)
    print('Trainable parameter count: {:,d} -> {:.2f} MB'.format(param_count, param_count*32/8/(2**20)))
    x = torch.randn((4, 2, 300, 161), dtype=torch.float32)
    y = net(x)
    print('{} -> {}'.format(x.shape, y.shape))

