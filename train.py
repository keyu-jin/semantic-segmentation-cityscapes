import torch
from torch import nn
from torch import optim
from model import UNet, UNet_ASPP,UNetPlusPlus, DeepLabV3Plus,SegNet,WSegNet,WSegNetplus,UNetPlusPlus
from visualize_dataset import load_dataset
from test import calculate_accuracy, calculate_iou, calculate_confusion_matrix
import numpy as np
import time
import json
import os
torch.manual_seed(17)

def load_hyperparameters(json_file_path, model_name):
    # 读取JSON文件
    with open(json_file_path, 'r') as json_file:
        hyperparameters = json.load(json_file)
    
    # 获取指定模型的超参数
    model_hyperparams = hyperparameters.get(model_name)
    
    if model_hyperparams is not None:
        return model_hyperparams
    else:
        raise ValueError(f"Model '{model_name}' not found in hyperparameters file.")

# 深度监督的权重
weights = np.array([1, 1, 1, 1])
weights = weights / np.sum(weights)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, device='cuda', lr_scheduler=None,log_path='./record/model.txt'):
    with open(log_path,'a') as f:
            f.write('\n')
    """训练模型"""
    # 记录开始时间
    start_time = time.time()
    loss_list = []
    train_acc_list = []
    train_iou_list = []
    val_acc_list = []
    val_iou_list = []

    train_acc_list = []
    train_iou_list = []
    val_acc_list = []
    val_iou_list = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_pixels = 0
        total_intersection = 0
        total_union = 0
        
        epoch_begin_time = time.time()
        total_correct = 0
        total_pixels = 0
        total_iou = 0
        total_intersection = 0
        total_union = 0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, musk = data[0].to(device), data[1].to(device)
            musk = musk.long()
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # 正向传播
            outputs = model(inputs)
            if model.name == 'unetpp':
                outputs, ds1, ds2, ds3 = model(inputs)
                loss = 0
                loss += weights[0] * criterion(outputs, musk)  # 最后一层的损失
                loss += weights[1] * criterion(ds1, musk)  # 第一层深度监督的损失
                loss += weights[2] * criterion(ds2, musk)  # 第二层深度监督的损失
                loss += weights[3] * criterion(ds3, musk)  # 第三层深度监督的损失
            else:
                loss = criterion(outputs, musk)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 累加损失值
            running_loss += loss.item()
            # 计算像素准确率和交并比
            _, predicted = torch.max(outputs, 1)
            total_correct, total_pixels = calculate_accuracy(predicted, musk)
            intersection, union = calculate_iou(predicted, musk, outputs.shape[1])
            total_intersection += intersection
            total_union += union

        # 更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step() 
        epoch_end_time = time.time()
        # 计算每个epoch的平均损失、像素准确率和交并比
        loss_list.append(running_loss / len(train_loader))
        train_acc = total_correct / total_pixels
        train_acc_list.append(train_acc)
        train_iou = total_intersection / total_union if total_union != 0 else 0
        train_iou_list.append(train_iou)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.3f}, Train Acc: {train_acc:.3f}, Train IoU: {train_iou:.3f}, cost {(epoch_end_time - epoch_begin_time)//60} min')
        
        """验证模型"""
        model.eval()
        total_correct = 0
        total_pixels = 0
        total_intersection = 0
        total_union = 0
        with torch.no_grad():
            for i, (img,musk) in enumerate(val_loader,0):
                img, musk = img.to(device), musk.to(device)
                musk = musk.long()
                if model.name == 'unetpp':
                    outputs,_,_,_ = model(img)
                else:
                    outputs = model(img)

                # 计算像素准确率和交并比
                _, predicted = torch.max(outputs, 1)
                total_correct, total_pixels = calculate_accuracy(predicted, musk)
                intersection, union = calculate_iou(predicted, musk, outputs.shape[1])
                total_intersection += intersection
                total_union += union

            # 计算每个epoch的平均损失、像素准确率和交并比
            val_acc = total_correct / total_pixels
            val_acc_list.append(val_acc)
            val_iou = total_intersection / total_union if total_union != 0 else 0
            val_iou_list.append(val_iou)
            print(f'\tValidation Acc: {val_acc:.3f}, Validation IoU: {val_iou:.3f}')
            if (val_acc >= np.max(val_acc_list)):  # 保存在验证集上效果最好的模型
                torch.save(model.state_dict(), f'./{model.name}.pth')
        
        """记录模型训练信息"""
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_path,'a') as f:
            f.write(f'\nEpoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.3f}, Train Acc: {train_acc:.3f}, Train IoU: {train_iou:.3f},Validation Acc: {val_acc:.3f}, Validation IoU: {val_iou:.3f}')

    end_time = time.time()
    print(f"Training costs {(end_time - start_time) / 3600 :.2f} hour")
    return loss_list, train_acc_list, train_iou_list, val_acc_list, val_iou_list

if __name__ == "__main__":
    MAX_EPOCHS = 20
    Batch_size = 8

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)
    model = WSegNetplus(device=device).to(device)
    model_weights_path = f'./{model.name}.pth'
    log_path = f'./record/{model.name}.txt'

    #加载训练超参数
    json_file_path = './hyperparameters.json'
    hyperparams = load_hyperparameters(json_file_path, model.name)
    LR = hyperparams['learning_rate']
    step_size = hyperparams['step_size']
    gamma = hyperparams['gamma']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loader, val_loader = load_dataset(batch_size=Batch_size)

    try:
        # 尝试加载模型权重
        model.load_state_dict(torch.load(model_weights_path, map_location=device,weights_only=True))
        print(f"Loaded model weights from {model_weights_path}")
    except FileNotFoundError:
        # 如果权重文件不存在，则打印消息并继续
        print(f"No model weights found at {model_weights_path}. Starting training from scratch.")

    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=MAX_EPOCHS, device=device, lr_scheduler=lr_scheduler,log_path=log_path)