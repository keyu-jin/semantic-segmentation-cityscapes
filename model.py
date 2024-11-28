import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation_models
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights,DeepLabV3_ResNet101_Weights
import numpy as np
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from wtconv import WTConv2d

# 辅助函数和类定义
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_down(nn.Module):
    def __init__(self, in_channels, out_channels,wavelet=False):
        super(DoubleConv_down, self).__init__()
        if wavelet:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConv_up(nn.Module):
    def __init__(self, in_channels, out_channels,wavelet = False):
        super(DoubleConv_up, self).__init__()
        self.wavelet = wavelet
        if self.wavelet:
            self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
            self.attention = SelfAttention(in_channels)
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.wavelet:
            x = self.double_conv(x)
            return x
        else:
            return self.double_conv(x)

class TripleConv_down(nn.Module):
    def __init__(self,in_channels,out_channels,wavelet=False):
        super(TripleConv_down,self).__init__()
        self.wavelet = wavelet
        if self.wavelet:
            self.triple_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                ResConv(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.attention = SelfAttention(out_channels)
            self.aspp = ASPP(out_channels,out_channels,8)
            self.skip_connection = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
        else:
            self.triple_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    def forward(self,x):
        if self.wavelet:
            idenity = self.skip_connection(x)
            x = self.triple_conv(x)
            x = self.attention(x)
            x = x + idenity
            return x
        else:
            return self.triple_conv(x)
    
class TripleConv_up(nn.Module):
    def __init__(self,in_channels,out_channels,wavelet=False):
        super(TripleConv_up,self).__init__()
        self.wavelet = wavelet
        if self.wavelet:
            self.triple_conv = nn.Sequential(
                ResConv(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.attention = SelfAttention(in_channels)
            self.aspp = ASPP(out_channels,out_channels,8)
            self.skip_connection = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)
        else:
            self.triple_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    def forward(self,x):
        if self.wavelet:
            idenity = self.skip_connection(x)
            x = self.attention(x)
            x = self.triple_conv(x)
            x = x + idenity
            return x
        else:
            return self.triple_conv(x)

class ResConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1,wavelet=False):
        super(ResConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if wavelet:
            self.conv2 = nn.Sequential(
                WTConv2d(out_channels,out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.skip_connection = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1)

    def forward(self,x):
        identity = self.skip_connection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity

        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        Q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)  # (B, N, C')
        K = self.key(x).view(B, -1, W * H)  # (B, C', N)
        V = self.value(x).view(B, -1, W * H)  # (B, C, N)

        attention = self.softmax(torch.bmm(Q, K) / (C ** 0.5))  # (B, N, N)
        out = torch.bmm(V, attention.permute(0, 2, 1)).view(B, C, W, H)

        return out + x  # Residual connection
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels,output_stride):
        super(ASPP, self).__init__()
        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are supported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class Down_Res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_Res, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=2,stride=2),
            ResConv(out_channels, out_channels),
            ResConv(out_channels,out_channels)
        )
        
    def forward(self, x):
        x = self.maxpool_conv(x)

        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels,out_channels)
        )
        self.combine = nn.Conv2d(2*out_channels,out_channels,kernel_size=1,stride=1)
        
    def forward(self, x):
        x = self.maxpool_conv(x)

        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv(x1)
        diffY = np.abs(x1.size()[2] - x2.size()[2])#高的差
        diffX = np.abs(x1.size()[3] - x2.size()[3])#宽的差

        x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet_ASPP(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 20, bilinear: bool = True):
        super(UNet_ASPP, self).__init__()
        self.name = 'unet_aspp'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器（下采样）
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down_Res(64, 128)
        self.down2 = Down_Res(128, 256)
        self.down3 = Down_Res(256, 512)
        self.down4 = Down_Res(512, 1024)

        self.aspp = ASPP(1024,1024,output_stride=8)
        
        # 解码器（上采样）
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.aspp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 20, bilinear: bool = True):
        super(UNet, self).__init__()
        self.name = 'unet'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器（下采样）
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.drop = nn.Dropout2d(0.5)

        # 解码器（上采样）
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.drop(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 20, bilinear: bool = True):
        super(UNetPlusPlus, self).__init__()
        self.name = 'unetpp'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 编码器（下采样）
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器（上采样）
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

        # 额外的跳跃连接
        self.extra_conv1 = DoubleConv(512, 256)
        self.extra_conv2 = DoubleConv(256, 128)
        self.extra_conv3 = DoubleConv(128, 64)

        # 深度监督的1x1卷积层
        self.conv_ds = nn.Conv2d(64, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x: torch.Tensor):
        x0_0 = self.inc(x)  # 输入层
        x1_0 = self.down1(x0_0)  # 第一层下采样
        x2_0 = self.down2(x1_0)  # 第二层下采样
        x3_0 = self.down3(x2_0)  # 第三层下采样
        x4_0 = self.down4(x3_0)  # 第四层下采样

        # 解码器和额外的跳跃连接
        x4_0 = self.drop(x4_0)
        x3_1 = self.up1(x4_0, x3_0)  # 倒数第一层
        x3_0 = self.drop(x3_0)
        x2_1 = self.up2(x3_0,x2_0)# 倒数第二层
        x = self.extra_conv1(torch.cat([x2_1, x2_0], dim=1))  # 额外的跳跃连接
        x2_2 = self.up2(x3_1,x)
        x2_0 = self.drop(x2_0)
        x1_1 = self.up3(x2_0,x1_0)#倒数第三层
        x = self.extra_conv2(torch.cat([x1_1, x1_0], dim=1))  # 额外的跳跃连接
        x1_2 = self.up3(x2_1,x)
        x = self.extra_conv2(torch.cat([x1_2, x1_1], dim=1))  # 额外的跳跃连接
        x = self.extra_conv2(torch.cat([x, x1_0], dim=1))  # 额外的跳跃连接
        x1_3 = self.up3(x2_2,x)
        x0_1 = self.drop(x1_0)
        x0_1 = self.up4(x1_0,x0_0)#倒数第四层
        ds1 = self.conv_ds(x0_1)
        x = self.extra_conv3(torch.cat([x0_1, x0_0], dim=1))  # 额外的跳跃连接
        x0_2 = self.up4(x1_1,x)
        ds2 = self.conv_ds(x0_2)
        x = self.extra_conv3(torch.cat([x0_2, x0_1], dim=1))  # 额外的跳跃连接
        x = self.extra_conv3(torch.cat([x, x0_0], dim=1))  # 额外的跳跃连接
        x0_3 = self.up4(x1_2,x)
        ds3 = self.conv_ds(x0_3)
        x = self.extra_conv3(torch.cat([x0_3, x0_2], dim=1))  # 额外的跳跃连接
        x = self.extra_conv3(torch.cat([x, x0_1], dim=1))  # 额外的跳跃连接
        x = self.extra_conv3(torch.cat([x, x0_0], dim=1))  # 额外的跳跃连接
        x0_4 = self.up4(x1_3,x)

        logits = self.outc(x0_4)

        return logits, ds1, ds2, ds3
    
    def outputs_vote(self,outputs):
        num_classes = outputs[1].shape[1]  # 每个输出的shape为[batch_size, num_classes, height, width]
        num_layers = len(outputs)

        # 初始化一个数组来存储每个类别的计数
        vote_counts = torch.zeros_like(outputs[1], dtype=torch.long)

        # 遍历每一层的输出
        for output in outputs:
            # 获取每个像素点的类别预测
            _, predicted = torch.max(output, dim=1)
            
            # 对每个类别进行计数
            for i in range(num_classes):
                vote_counts[:, i, :, :] += (predicted == i).long()

        # 找出每个像素点出现次数最多的类别
        _, voted_predictions = torch.max(vote_counts, dim=1)

        # 如果所有层的输出结果都不同，则选择最后一层的输出结果
        same_as_last_layer = (voted_predictions == outputs[0].argmax(dim=1)).float()
        mask = same_as_last_layer.sum(dim=1) == num_layers  # 如果所有层都相同，则mask为True

        # 应用mask，如果所有层都相同，则选择最后一层的结果，否则选择投票结果
        final_predictions = torch.where(mask.unsqueeze(1), outputs[-1].argmax(dim=1), voted_predictions)
        return final_predictions

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes:int=20):
        super(DeepLabV3Plus, self).__init__()
        self.name = 'deeplabv3p'
        # 加载预训练的DeepLabV3+模型，使用weights参数
        self.model = segmentation_models.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        
        # 替换最后的分类层以适应Cityscapes数据集的类别数
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, dilation=1)
        
    def forward(self, x):
        return self.model(x)['out']  # 'out' 是包含实际输出的键

class Seg_Encoder(nn.Module):
    def __init__(self,in_channels:int=3):
        super(Seg_Encoder,self).__init__()
        self.stage_1 = DoubleConv_down(in_channels,64)
        self.stage_2 = DoubleConv_down(64,128)
        self.stage_3 = TripleConv_down(128,256)  
        self.stage_4 = TripleConv_down(256,512)  
        self.stage_5 = TripleConv_down(512,512)     
        
    def forward(self, x):
        #用来保存各层的池化索引
        pool_indices = []
        x = x.float()
        
        x = self.stage_1(x)
        #pool_indice_1保留了第一个池化层的索引
        x, pool_indice_1 = nn.MaxPool2d( 2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_1)
        
        x = self.stage_2(x)
        x, pool_indice_2 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_2)
        
        x = self.stage_3(x)
        x, pool_indice_3 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_3)   
        
        x = self.stage_4(x)
        x, pool_indice_4 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_4)
        
        x = self.stage_5(x)
        x, pool_indice_5 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_5)
        
        return x, pool_indices
    
class SegNet(nn.Module):
    def __init__(self,in_channels:int=3, num_classes:int=20):
        super(SegNet, self).__init__()
        self.name = 'segnet'
        self.encoder = Seg_Encoder(in_channels)
       #上采样 从下往上, 1->2->3->4->5
        self.upsample_1 = TripleConv_up(512,512)
        self.upsample_2 = TripleConv_up(512,256)
        self.upsample_3 = TripleConv_up(256,128)
        self.upsample_4 = DoubleConv_up(128,64)
        self.upsample_5 = DoubleConv_up(64,num_classes)
        self.drop = nn.Dropout2d(0.4)   
        
    def forward(self, x):
        x, pool_indices = self.encoder(x)
        x = self.drop(x)
        #池化索引上采样
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[4])
        x = self.upsample_1(x)
        
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[3])
        x = self.upsample_2(x) 
        
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[2])
        x = self.upsample_3(x)
        
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[1])
        x = self.upsample_4(x)
        
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[0])
        x = self.upsample_5(x)
        
        return x

class WSeg_Encoder(Seg_Encoder):
    def __init__(self,in_channels:int=3,wavename:str='haar',device='cuda0'):
        super(WSeg_Encoder,self).__init__()
        self.device = device
        self.dwt = DWT_2D(wavename = wavename,device=self.device)
        
    def forward(self, x):
        x = x.float()
        x = self.stage_1(x)
        LL1,LH1,HL1,HH1 = self.dwt(x)
        x = self.stage_2(LL1)
        LL2,LH2,HL2,HH2 = self.dwt(x)
        x = self.stage_3(LL2)
        LL3,LH3,HL3,HH3 = self.dwt(x)
        x = self.stage_4(LL3)
        LL4,LH4,HL4,HH4 = self.dwt(x)
        x = self.stage_5(LL4)
        LL5,LH5,HL5,HH5 = self.dwt(x)
           
        return LL5, [(LH1,HL1,HH1), (LH2,HL2,HH2,), (LH3,HL3,HH3,), (LH4,HL4,HH4,), (LH5,HL5,HH5,)]

class WSegNet(SegNet):
    def __init__(self,in_channels:int=3, num_classes:int=20,wavename:str='haar',device='cuda0'):
        super(WSegNet, self).__init__()
        self.name = 'wsegnet'
        self.device = device
        self.encoder = WSeg_Encoder(in_channels,wavename,self.device)
        self.idwt = IDWT_2D(wavename = wavename,device=self.device)
        self.upsample_5 = DoubleConv_up(64,num_classes) 
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x, wavelet_componets = self.encoder(x)
        x = self.drop(x)
        x = self.idwt(x,*wavelet_componets[4])
        x = self.upsample_1(x)
        x = self.idwt(x,*wavelet_componets[3])
        x = self.upsample_2(x)
        x = self.idwt(x,*wavelet_componets[2])
        x = self.upsample_3(x)
        x = self.idwt(x,*wavelet_componets[1])
        x = self.upsample_4(x)
        x = self.idwt(x,*wavelet_componets[0])
        x = self.upsample_5(x)

        return x

class WSegNetplus(WSegNet):
    def __init__(self,in_channels:int=3, num_classes:int=20,wavename:str='haar',device='cuda0'):
        super(WSegNetplus, self).__init__()
        self.name = 'wsegnetp'
        self.device = device
        #Encoder
        self.stage_1 = DoubleConv_down(in_channels,64,wavelet=True)
        self.stage_2 = DoubleConv_down(64,128,wavelet=True)
        self.stage_3 = TripleConv_down(128,512,wavelet=True)
        self.stage_4 = TripleConv_down(512,1024,wavelet=True)
        self.stage_5 = TripleConv_down(1024,1024,wavelet=True)
        self.dwt = DWT_2D(wavename = wavename,device=self.device) 

        #Decoder
        self.idwt = IDWT_2D(wavename = wavename,device=self.device)
        self.upsample_1 = TripleConv_up(1024,1024,wavelet=True)
        self.upsample_2 = TripleConv_up(1024,512,wavelet=True)
        self.upsample_3 = TripleConv_up(512,128,wavelet=True)
        self.upsample_4 = DoubleConv_up(128,64,wavelet=True)
        self.upsample_5 = DoubleConv_up(64,num_classes,wavelet=True)
        self.drop = nn.Dropout2d(0.6)

        self.unet_connection4 = nn.Conv2d(1024*2,1024,kernel_size=1,stride=1)
        self.unet_connection3 = nn.Conv2d(512*2,512,kernel_size=1,stride=1)
        self.unet_connection2 = nn.Conv2d(128*2,128,kernel_size=1,stride=1)
        self.unet_connection1 = nn.Conv2d(64*2,64,kernel_size=1,stride=1)

    def forward(self, x):
        x = x.float()
        x = self.stage_1(x)
        LL1,LH1,HL1,HH1 = self.dwt(x)
        x = self.stage_2(LL1)
        LL2,LH2,HL2,HH2 = self.dwt(x)
        x = self.stage_3(LL2)
        LL3,LH3,HL3,HH3 = self.dwt(x)
        x = self.stage_4(LL3)
        LL4,LH4,HL4,HH4 = self.dwt(x)
        x = self.stage_5(LL4)
        LL5,LH5,HL5,HH5 = self.dwt(x)

        x = self.drop(LL5)
    
        x = self.idwt(x,*(LH5,HL5,HH5,))
        x = self.upsample_1(x)
        
        x = self.unet_connection4(torch.cat((x,LL4),dim=1))
        x = self.idwt(x,*(LH4,HL4,HH4,))
        x = self.upsample_2(x)

        x = self.unet_connection3(torch.cat((x,LL3),dim=1))
        x = self.idwt(x,*(LH3,HL3,HH3,))
        x = self.upsample_3(x)

        x = self.unet_connection2(torch.cat((x,LL2),dim=1))
        x = self.idwt(x,*(LH2,HL2,HH2,))
        x = self.upsample_4(x)

        x = self.unet_connection1(torch.cat((x,LL1),dim=1))
        x = self.idwt(x,*(LH1,HL1,HH1))
        x = self.upsample_5(x)

        return x

# Example usage
if __name__ == "__main__":
    device = torch.device('cpu')
    model = WSegNetplus(device=device).to(device)
    print(model)
    # Example input tensor
    x = torch.randn(8, 3, 128, 256)
    output = model(x)
    print(output.shape)  # Should be [8, 20, 128, 256]
    #_, predicted = torch.max(output, 1)
    #print(predicted.shape)