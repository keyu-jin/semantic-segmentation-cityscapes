import os
from PIL import Image
import albumentations as A
import numpy as np

def resize_image(input_path_list, output_path_list, augmentation = False, size=(256, 128)):
    for i,(input_path,output_path) in enumerate(input_path_list,output_path_list):
        img = Image.open(input_path)
        resized_img = img.resize(size, Image.NEAREST)
        resized_img.save(output_path)            


def return_destination_folder_path_list(source_folder_path,new_folder_path,musk=False):
    src_path_list = []
    dst_path_list = []
    for file_name in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, file_name)
        for image in os.listdir(file_path):
            if musk == True:
                if image.endswith('labelTrainIds.png'):
                    src_path_list.append(os.path.join(file_path, image))
                    dst_path_list.append(os.path.join(new_folder_path, image))
            else:
                src_path_list.append(os.path.join(file_path, image))
                dst_path_list.append(os.path.join(new_folder_path, image))

            #print(src_path_list)
            #print(dst_path_list)
    return src_path_list,dst_path_list

#  移动并压缩图像
def move_and_compress(src_img_path_list, dst_img_path_list,
                      src_mask_path_list, dst_mask_path_list):
    resize_size=(128, 256)
    for src_img_path, dst_img_path, src_mask_path, dst_mask_path in zip(src_img_path_list, dst_img_path_list, src_mask_path_list, dst_mask_path_list):
        #读取图像和掩码
        image = np.array(Image.open(src_img_path).convert("RGB"))
        mask = np.array(Image.open(src_mask_path))

        # 定义变换链
        transforms = A.Compose([
            A.Resize(resize_size[0], resize_size[1]),  # 随机调整大小到目标尺寸
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2)
        ], bbox_params=None, keypoint_params=None)

        # 应用变换
        transformed = transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        # 保存变换后的图像和掩码
        Image.fromarray(transformed_image).save(dst_img_path)
        Image.fromarray(transformed_mask).save(dst_mask_path)

def image_enhancement(src_img_path_list, dst_img_path_list,
                      src_mask_path_list, dst_mask_path_list):
    resize_size=(128, 256)
    for src_img_path, dst_img_path, src_mask_path, dst_mask_path in zip(src_img_path_list, dst_img_path_list, src_mask_path_list, dst_mask_path_list):
        #读取图像和掩码
        image = np.array(Image.open(src_img_path).convert("RGB"))
        mask = np.array(Image.open(src_mask_path))

        # 定义变换链
        transforms = A.Compose([
            A.RandomCrop(width=512,height=256,p=0.5),
            A.Resize(resize_size[0], resize_size[1]),  # 随机调整大小到目标尺寸
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ], bbox_params=None, keypoint_params=None)

        # 应用变换
        transformed = transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        # 获取原始文件名并添加前缀"aug_"
        base_name = os.path.basename(src_img_path)
        aug_img_name = "aug_" + base_name
        aug_mask_name = "aug_" + os.path.basename(src_mask_path)

        # 创建新的文件路径
        dst_img_path = os.path.join(os.path.dirname(dst_img_path), aug_img_name)
        dst_mask_path = os.path.join(os.path.dirname(dst_mask_path), aug_mask_name)

        # 保存变换后的图像和掩码
        Image.fromarray(transformed_image).save(dst_img_path)
        Image.fromarray(transformed_mask).save(dst_mask_path)

if __name__ == "__main__":
    # 数据集路径
    dataset_path = r"E:\Desktop\MachineLearning\assignment4\leftImg8bit_trainvaltest\leftImg8bit"
    mask_path = r"E:\Desktop\MachineLearning\assignment4\gtFine_trainvaltest\gtFine"
    compressed_dataset_path = r"E:\Desktop\MachineLearning\assignment4\semantic-segmentation-cityscapes\data"

    # 原始的train, valid文件夹路径
    train_source_img_path = os.path.join(dataset_path, 'train')
    val_source_img_path = os.path.join(dataset_path, 'val')
    test_source_img_path = os.path.join(dataset_path, 'test')

    train_source_musk_path = os.path.join(mask_path, 'train')
    val_source_musk_path = os.path.join(mask_path, 'val')
    test_source_musk_path = os.path.join(mask_path, 'test')

    # 创建train,valid的文件夹
    train_target_path = os.path.join(compressed_dataset_path, 'train')
    val_target_path = os.path.join(compressed_dataset_path, 'val')
    test_target_path = os.path.join(compressed_dataset_path, 'test')

    # 创建文件夹，如果已存在则不抛出异常
    os.makedirs(train_target_path, exist_ok=True)
    os.makedirs(val_target_path, exist_ok=True)
    os.makedirs(test_target_path, exist_ok=True)

    #整理训练数据集
    src_img_list,dst_img_list = return_destination_folder_path_list(train_source_img_path,train_target_path)
    src_musk_list,dst_musk_list = return_destination_folder_path_list(train_source_musk_path,train_target_path,musk=True)
    print(f"length of train dataset {len(src_musk_list)}")
    move_and_compress(src_img_list,dst_img_list,src_musk_list,dst_musk_list)
    #image_enhancement(src_img_list,dst_img_list,src_musk_list,dst_musk_list)

    #整理验证数据集
    src_img_list,dst_img_list = return_destination_folder_path_list(val_source_img_path,val_target_path)
    src_musk_list,dst_musk_list = return_destination_folder_path_list(val_source_musk_path,val_target_path,musk=True)
    print(f"length of validation dataset {len(src_musk_list)}")
    move_and_compress(src_img_list,dst_img_list,src_musk_list,dst_musk_list)

    #整理测试数据集
    src_img_list,dst_img_list = return_destination_folder_path_list(test_source_img_path,test_target_path)
    src_musk_list,dst_musk_list = return_destination_folder_path_list(test_source_musk_path,test_target_path,musk=True)
    print(f"length of test dataset {len(src_musk_list)}")
    move_and_compress(src_img_list,dst_img_list,src_musk_list,dst_musk_list)
    
    