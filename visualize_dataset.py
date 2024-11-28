# 导入库
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


# 自定义数据集CityScapesDataset
class CityScapesDataset(Dataset):
    def __init__(self, dataset_folder_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        ])
        self.data_dir = dataset_folder_path
        self.ids = os.listdir(self.data_dir)
        self.images_fps = [os.path.join(self.data_dir, image_id) 
                           for image_id in self.ids
                           if 'leftImg8bit' in image_id]
        self.masks_fps = [os.path.join(self.data_dir, musk_id) 
                          for musk_id in self.ids
                          if 'labelTrainIds' in musk_id]

    def __getitem__(self, i):
        # 读取图像和掩码
        image = Image.open(self.images_fps[i]).convert('RGB')
        mask = Image.open(self.masks_fps[i])
        
        # 应用转换
        image = self.transform(image)
        mask = np.array(mask)
        # 将忽视区域（255）转换为-1
        mask[mask == 255] = -1
        # 归一化掩码，使得类别标签均匀分布在0到num_classes-1之间
        num_classes = 19  # 有19个类别
        mask = (mask + 1)  # +1是为了将-1映射到0，其余标签映射到1到19

        return image,mask

    def __len__(self):
        return len(self.images_fps)
    
def load_dataset(batch_size=8):
    # 设置数据集路径
    train_dir = r".\data\train"
    valid_dir = r".\data\val"

    train_dataset = CityScapesDataset(train_dir)
    val_dataset = CityScapesDataset(valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=1)

    return train_loader,val_loader

def visulize_img(img,label):
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow((img[0,:,:,:].moveaxis(0,2)))
    plt.subplot(222)
    plt.imshow(label[0,:,:],cmap='cividis')
    
    plt.subplot(223)
    plt.imshow((img[6,:,:,:].moveaxis(0,2)))
    plt.subplot(224)
    plt.imshow(label[6,:,:],cmap='cividis') 
    plt.show()

if __name__ == "__main__":
    train_loader,_ = load_dataset(batch_size=8)
    print(len(train_loader))
    for i, (img,musk) in enumerate(train_loader,0):
        print(musk.shape)
        print(img.shape)
        print(musk[0,:10,:10])

        visulize_img(img,musk)

        break