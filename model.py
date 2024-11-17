import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

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
    
class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 19, bilinear: bool = True):
        super(UNet, self).__init__()
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

    def forward(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 19, bilinear: bool = True):
        super(UNetPlusPlus, self).__init__()
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

    def forward(self, x: torch.Tensor):
        x0_0 = self.inc(x)  # 输入层
        x1_0 = self.down1(x0_0)  # 第一层下采样
        x2_0 = self.down2(x1_0)  # 第二层下采样
        x3_0 = self.down3(x2_0)  # 第三层下采样
        x4_0 = self.down4(x3_0)  # 第四层下采样

        # 解码器和额外的跳跃连接
        x3_1 = self.up1(x4_0, x3_0)  # 倒数第一层
        x2_1 = self.up2(x3_0,x2_0)# 倒数第二层
        x = self.extra_conv1(torch.cat([x2_1, x2_0], dim=1))  # 额外的跳跃连接
        x2_2 = self.up2(x3_1,x)
        x1_1 = self.up3(x2_0,x1_0)#倒数第三层
        x = self.extra_conv2(torch.cat([x1_1, x1_0], dim=1))  # 额外的跳跃连接
        x1_2 = self.up3(x2_1,x)
        x = self.extra_conv2(torch.cat([x1_2, x1_1], dim=1))  # 额外的跳跃连接
        x = self.extra_conv2(torch.cat([x, x1_0], dim=1))  # 额外的跳跃连接
        x1_3 = self.up3(x2_2,x)
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

# Example usage
if __name__ == "__main__":
    model = UNetPlusPlus()
    print(model)
    # Example input tensor
    x = torch.randn(8, 3, 128, 256)
    output = model(x)
    print(output.shape)  # Should be [8, 19, 128, 256]