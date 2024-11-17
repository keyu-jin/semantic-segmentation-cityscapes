import torch
from torch import nn
from torch import optim
from model import UNet,UNetPlusPlus
from visualize_dataset import load_dataset
import numpy as np
import time

# 深度监督的权重
weights = np.array([1, 0.5, 0.5, 0.5])
weights = weights/np.sum(weights)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, device='cuda',name='unet'):
    # 记录开始时间
    start_time = time.time()
    loss_list = []
    train_acc_list = []
    train_iou_list = []
    val_acc_list = []
    val_iou_list = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_pixels = 0
        total_iou = 0
        total_intersection = 0
        total_union = 0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            labels = labels.squeeze(1)  # 去除不必要的通道维度
            labels = labels.long()  # 转换标签为 Long 类型

            if name == 'unetpp':
                outputs, ds1, ds2, ds3 = model(inputs)
                loss = 0
                loss += weights[0] * criterion(outputs, labels)  # 最后一层的损失
                loss += weights[1] * criterion(ds1, labels)  # 第一层深度监督的损失
                loss += weights[2] * criterion(ds2, labels)  # 第二层深度监督的损失
                loss += weights[3] * criterion(ds3, labels)  # 第三层深度监督的损失
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 累加损失值
            running_loss += loss.item()

            # 计算像素准确率和交并比
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()

            # 计算交并比
            for cls in range(outputs.shape[1]):
                pred_inds = (predicted == cls).view(-1)
                true_inds = (labels == cls).view(-1)
                intersection = (pred_inds & true_inds).sum().item()
                union = (pred_inds | true_inds).sum().item()
                total_iou += intersection / union if union != 0 else 0
                total_intersection += intersection
                total_union += union

        # 计算每个epoch的平均损失、像素准确率和交并比
        loss_list.append(running_loss / len(train_loader))
        train_acc = total_correct / total_pixels
        train_acc_list.append(train_acc)
        train_iou = total_iou / (len(train_loader) * outputs.shape[1])
        train_iou_list.append(train_iou)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.3f}, Train Acc: {train_acc:.3f}, Train IoU: {train_iou:.3f}')

        if (np.mod(epoch,9) == 0):
            # 验证集上的评估 每十epoch评估一次
            model.eval()
            val_correct = 0
            val_pixels = 0
            val_iou = 0
            val_intersection = 0
            val_union = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels = labels.squeeze(1)
                    labels = labels.long()
                    if name == 'unetpp':
                        outputs,_,_,_ = model(inputs)
                    else:
                        outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_pixels += labels.numel()

                    # 计算交并比
                    for cls in range(outputs.shape[1]):
                        pred_inds = (predicted == cls).view(-1)
                        true_inds = (labels == cls).view(-1)
                        intersection = (pred_inds & true_inds).sum().item()
                        union = (pred_inds | true_inds).sum().item()
                        val_iou += intersection / union if union != 0 else 0
                        val_intersection += intersection
                        val_union += union

            val_acc = val_correct / val_pixels
            val_acc_list.append(val_acc)
            val_iou = val_iou / (len(val_loader) * outputs.shape[1])
            val_iou_list.append(val_iou)
            print(f'Validation Acc: {val_acc:.3f}, Validation IoU: {val_iou:.3f}')

            # 保存模型
            torch.save(model.state_dict(), f'./{name}.pth')
        
    end_time = time.time()
    print(f"Training costs {(end_time-start_time)//60} min")
    return loss_list, train_acc_list, train_iou_list, val_acc_list, val_iou_list

if __name__ == "__main__":
    MAX_EPOCHS = 20
    Batch_size = 8
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetPlusPlus(in_channels=3, num_classes=19).to(device)
    name = 'unetpp'
    model_weights_path = f'./{name}.pth'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader = load_dataset(batch_size=Batch_size)

    try:
        # 尝试加载模型权重
        model.load_state_dict(torch.load(model_weights_path, map_location=device,weights_only=True))
        print(f"Loaded model weights from {model_weights_path}")
    except FileNotFoundError:
        # 如果权重文件不存在，则打印消息并继续
        print(f"No model weights found at {model_weights_path}. Starting training from scratch.")

    train_model(model, criterion, optimizer, 
                train_loader, val_loader,num_epochs=MAX_EPOCHS,
                device=device,name='unetpp')
