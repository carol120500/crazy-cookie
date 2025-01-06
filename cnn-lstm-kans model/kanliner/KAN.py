from torch import nn
import torch.nn.functional as F
from Kan_Convolution.KANConv import KAN_Convolutional_Layer
from Kan_Convolution.KANLinear import KANLinear

class KAN_Convolution_Network(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        # 创建一个KAN_Convolutional_Layer实例作为第一层卷积层
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5, # 包含了5个卷积操作 
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        self.flat = nn.Flatten() 
        # self.fc = nn.Linear(625, 10)
        self.kan1 = KANLinear(
            625,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x) 
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
    
# 项目讲解链接：https://www.bilibili.com/video/BV1Nw4m1D75E/?spm_id_from=333.999.0.0&vd_source=6528929ff3772e61e9f5baf9b8ab1e64