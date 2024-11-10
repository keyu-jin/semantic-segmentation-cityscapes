import os
import shutil
from PIL import Image

# 定义一个函数来压缩图像
def resize_image(input_path, output_path, size=(256, 128)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.NEAREST)
        resized_img.save(output_path)

# 数据集路径
dataset_path = r"E:\Desktop\MachineLearning\assignment4\leftImg8bit_trainvaltest\leftImg8bit"
compressed_dataset_path = r"E:\Desktop\MachineLearning\assignment4\semantic-segmentation-cityscapes\data"

# 原始的train, valid文件夹路径
train_dataset_path = os.path.join(dataset_path, 'train')
val_dataset_path = os.path.join(dataset_path, 'val')
test_dataset_path = os.path.join(dataset_path, 'test')
# 创建train,valid的文件夹
train_images_path = os.path.join(compressed_dataset_path, 'cityscapes_train')
val_images_path = os.path.join(compressed_dataset_path, 'cityscapes_val')
test_images_path = os.path.join(compressed_dataset_path, 'cityscapes_test')

# 创建文件夹，如果已存在则不抛出异常
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

# -----------------移动并压缩图像-------------------------------------------------
for file_name in os.listdir(train_dataset_path):
    file_path = os.path.join(train_dataset_path, file_name)
    for image in os.listdir(file_path):
        src_path = os.path.join(file_path, image)
        dst_path = os.path.join(train_images_path, image)
        #shutil.copy(src_path, dst_path)
        resize_image(src_path, dst_path)

for file_name in os.listdir(val_dataset_path):
    file_path = os.path.join(val_dataset_path, file_name)
    for image in os.listdir(file_path):
        src_path = os.path.join(file_path, image)
        dst_path = os.path.join(val_images_path, image)
        #shutil.copy(src_path, dst_path)
        resize_image(src_path, dst_path)

for file_name in os.listdir(test_dataset_path):
    file_path = os.path.join(test_dataset_path, file_name)
    for image in os.listdir(file_path):
        src_path = os.path.join(file_path, image)
        dst_path = os.path.join(test_images_path, image)
        #shutil.copy(src_path, dst_path)
        resize_image(src_path, dst_path)

# 数据集路径
dataset_path = r"E:\Desktop\MachineLearning\assignment4\gtFine_trainvaltest\gtFine"
compressed_dataset_path = r"E:\Desktop\MachineLearning\assignment4\semantic-segmentation-cityscapes\data"

# 原始的train, valid文件夹路径
train_dataset_path = os.path.join(dataset_path, 'train')
val_dataset_path = os.path.join(dataset_path, 'val')
test_dataset_path = os.path.join(dataset_path, 'test')
# 创建train,valid的文件夹
train_images_path = os.path.join(compressed_dataset_path, 'cityscapes_19classes_train')
val_images_path = os.path.join(compressed_dataset_path, 'cityscapes_19classes_val')
test_images_path = os.path.join(compressed_dataset_path, 'cityscapes_19classes_test')

# 创建文件夹，如果已存在则不抛出异常
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

# -----------------移动并压缩文件---对于19类语义分割, 主需要原始图像中的labelIds结尾图片------------
for file_name in os.listdir(train_dataset_path):
    file_path = os.path.join(train_dataset_path, file_name)
    for image in os.listdir(file_path):
        if image.split('.png')[0][-13:] == "labelTrainIds":
            src_path = os.path.join(file_path, image)
            dst_path = os.path.join(train_images_path, image)
            #shutil.copy(src_path, dst_path)
            resize_image(src_path, dst_path)

for file_name in os.listdir(val_dataset_path):
    file_path = os.path.join(val_dataset_path, file_name)
    for image in os.listdir(file_path):
        if image.split('.png')[0][-13:] == "labelTrainIds":
            src_path = os.path.join(file_path, image)
            dst_path = os.path.join(val_images_path, image)
            #shutil.copy(src_path, dst_path)
            resize_image(src_path, dst_path)

for file_name in os.listdir(test_dataset_path):
    file_path = os.path.join(test_dataset_path, file_name)
    for image in os.listdir(file_path):
        if image.split('.png')[0][-13:] == "labelTrainIds":
            src_path = os.path.join(file_path, image)
            dst_path = os.path.join(test_images_path, image)
            #shutil.copy(src_path, dst_path)
            resize_image(src_path, dst_path)