import torch
from torch import nn
from torch import optim
from model import UNet
from visualize_dataset import load_dataset
import time

def train_model(model, criterion, optimizer, train_loader, num_epochs=25,device='cuda0'):
# 记录开始时间
    start_time = time.time()
    loss_list = []
    train_acc = []
    validation_acc = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            labels = labels.squeeze(1)  # 去除不必要的通道维度
            labels = labels.long()  # 转换标签为 Long 类型
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 累加损失值
            running_loss += loss.item()
            
        # 打印每个epoch的平均损失
        loss_list.append(running_loss / len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.3f}')
            

if __name__ == "__main__":
    MAX_EPOCHS = 20
    Batch_size = 8
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, num_classes=19).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, val_loader = load_dataset(batch_size=Batch_size)
    train_model(model, criterion, optimizer, train_loader, num_epochs=MAX_EPOCHS,device=device)