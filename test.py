import torch
from torch import nn
from model import UNet,UNetPlusPlus,DeepLabV3Plus,SegNet,WSegNet,WSegNetplus
from visualize_dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def calculate_accuracy(predicted, labels):
    """计算像素准确率"""
    correct = (predicted == labels).sum().item()
    pixels = labels.numel()
    return correct, pixels

def calculate_iou(predicted, labels, num_classes):
    """计算交并比"""
    intersection = 0
    union = 0
    for cls in range(num_classes):
        pred_inds = (predicted == cls).view(-1)
        true_inds = (labels == cls).view(-1)
        intersection += (pred_inds & true_inds).sum().item()
        union += (pred_inds | true_inds).sum().item()
    return intersection, union

def calculate_confusion_matrix(predicted, labels, num_classes,confusion_matrix):
    """计算混淆矩阵"""
    # 遍历每个类别
    for cls in range(num_classes):
        # 预测为cls的像素点
        pred_inds = (predicted == cls).view(-1)
        # 真实为cls的像素点
        true_inds = (labels == cls).view(-1)
        # 真实为cls且预测也为cls的像素点
        confusion_matrix[cls, cls] += (pred_inds & true_inds).sum().item()
        # 预测为cls但实际不是cls的像素点
        confusion_matrix[cls, :] += (pred_inds).sum().item()
        # 真实为cls但预测不是cls的像素点
        confusion_matrix[:, cls] += (true_inds).sum().item()
    
    
    return confusion_matrix

def normalize_confusion_matrix(confusion_matrix):
    # 将混淆矩阵转换为float类型，以便进行除法操作
    confusion_matrix = confusion_matrix.float()
    # 计算每行的和
    row_sums = confusion_matrix.sum(dim=1, keepdim=True)
    # 按行归一化
    normalized_confusion_matrix = confusion_matrix / row_sums
    return normalized_confusion_matrix

def plot_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('predicted label')
    plt.ylabel('True label')
    plt.title('confusion matrix')
    plt.show()

def test_model(model,val_loader,device):
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_intersection = 0
    total_union = 0
    num_classes = 20
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            img, labels = img.to(device), label.to(device)
            labels = labels.long()

            if model.name == 'unetpp':
                outputs,_,_,_ = model(img)
            else:
                outputs = model(img)
            _,predicted = torch.max(outputs,1)
            
            correct, pixels = calculate_accuracy(predicted, labels)
            total_correct += correct
            total_pixels += pixels
            intersection, union = calculate_iou(predicted, labels, outputs.shape[1])
            total_intersection += intersection
            total_union += union
            confusion_matrix = calculate_confusion_matrix(predicted,labels,outputs.shape[1],confusion_matrix)

        val_acc = total_correct / total_pixels if total_pixels != 0 else 0
        val_iou = total_intersection / total_union if total_union != 0 else 0
        print(f'Validation Acc: {val_acc:.3f}, Validation IoU: {val_iou:.3f}')
        confusion_matrix = normalize_confusion_matrix(confusion_matrix)
        class_list = range(20)
        plot_confusion_matrix(confusion_matrix,class_list)

def visualize_results(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        for i, (img,musk) in enumerate(val_loader,0):
            img, musk = img.to(device), musk.to(device)
            
            if model.name == 'unetpp':
                outputs,_,_,_ = model(img)
            else:
                outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            
            # 将预测和标签转换为numpy数组
            predicted_np = predicted.cpu().numpy()
            musk_np = musk.cpu().numpy()
            predicted_np = predicted_np[0]
            musk_np = musk_np[0]

            # 可视化输入图像
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            ax[0].imshow(img[0,0].cpu().numpy(),cmap='gray')
            ax[0].set_title('Input Image')
            
            # 可视化真实标签
            ax[1].imshow(musk_np,cmap='gray')
            ax[1].set_title('True Mask')
            
            # 可视化预测结果
            ax[2].imshow(predicted_np, cmap='gray')
            ax[2].set_title('Predicted Mask')
            
            ax[3].imshow(predicted_np, cmap='gray')
            ax[3].set_title('Error Mark')
            # 找出不一致的地方并用红色标出
            for j in range(predicted_np.shape[0]):
                for k in range(predicted_np.shape[1]):
                    if predicted_np[j, k] != musk_np[j, k]:
                        ax[3].add_patch(patches.Rectangle((k, j), 1, 1, linewidth=1, edgecolor='r', facecolor='none'))
            
            plt.show()
            break  # 只显示第一个测试样本的结果


if __name__ == "__main__":
    Batch_size = 8
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = DeepLabV3Plus().to(device)
    model_weights_path = f'./{model.name}.pth' 
    model.load_state_dict(torch.load(model_weights_path, map_location=device,weights_only=True))
    print(f"Loaded model weights from {model_weights_path}")
    
    # 加载测试集
    _, val_loader = load_dataset(batch_size=Batch_size)
    
    # 验证模型性能并可视化输出结果
    test_model(model,val_loader,device)
    visualize_results(model, val_loader, device)