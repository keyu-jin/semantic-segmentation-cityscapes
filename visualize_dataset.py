# 导入库
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
#from albumentations import Compose,Resize,HorizontalFlip,Normalize,RandomSnow
#from albumentations.pytorch.transforms import ToTensorV2

# 自定义数据集CityScapesDataset
class CityScapesDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.transform = transforms.Compose([
            #transforms.Resize((128, 256), interpolation=transforms.InterpolationMode.NEAREST),  # 使用最近邻插值法调整大小
            #transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
        self.mask_transform = transforms.Compose([
            #transforms.Resize((128, 256), interpolation=transforms.InterpolationMode.NEAREST),  # 使用最近邻插值法调整大小
            #transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor()  # 转换为张量
        ])
        self.ids = os.listdir(images_dir)
        self.ids2 = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids2]

    def __getitem__(self, i):
        # 读取图像和掩码
        image = Image.open(self.images_fps[i]).convert('RGB')
        mask = Image.open(self.masks_fps[i]).convert('L')
        
        # 应用转换
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image,mask

    def __len__(self):
        return len(self.ids)
    
def load_dataset(batch_size=8):
    # 设置数据集路径
    x_train_dir = r"data\cityscapes_train"
    y_train_dir = r"data\cityscapes_19classes_train"

    x_valid_dir = r"data\cityscapes_val"
    y_valid_dir = r"data\cityscapes_19classes_val"

    train_dataset = CityScapesDataset(
        x_train_dir,
        y_train_dir,
    )
    val_dataset = CityScapesDataset(
        x_valid_dir,
        y_valid_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,val_loader

if __name__ == "__main__":
    train_loader,_ = load_dataset(batch_size=8)
    for index, (img, label) in enumerate(train_loader):
        print(img.shape)
        print(label.shape)
        
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.imshow((img[0,:,:,:].moveaxis(0,2)))
        plt.subplot(222)
        plt.imshow(label[0,0,:,:],cmap='cividis')
        
        plt.subplot(223)
        plt.imshow((img[6,:,:,:].moveaxis(0,2)))
        plt.subplot(224)
        plt.imshow(label[6,0,:,:],cmap='cividis') 
        plt.show() 
        if index==0:
            break
