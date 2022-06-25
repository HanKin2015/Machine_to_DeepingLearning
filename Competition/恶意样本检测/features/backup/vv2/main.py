import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VGG import vgg16
from MalwareDataset import MalwareDataset, MalwareTrainDataset, MalwareTestDataset
from Configure import Configure
import sys
import pandas as pd
 
 
def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()   # 梯度清0
 
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
 
        y_pred = model(inputs)  # 前向传播
        loss = torch.nn.functional.cross_entropy(y_pred, labels.long())    # 计算损失
 
        if batch_idx % 100 == 99:
            print("epoch=%d, loss=%f" % (epoch, loss.item()))
 
        loss.backward()     # 反向传播
        optimizer.step()    # 梯度更新
 
# 用来测试Maling数据集
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum()
    acc = 1.0 * 100 * correct / total
    print('Accuracy on test set: %f %% [%d/%d]' % (acc, correct, total))
 
 
# 用来测试微软的公开数据集
def test():
    df = pd.DataFrame()
    with torch.no_grad():
        for inputs, file_name in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = F.softmax(outputs.data)
            # _, predicted = torch.max(outputs.data, dim=1)
            data_len = len(inputs)
            for i in range(data_len):
                dict_res = {"Id": file_name[i], "Prediction1": 0, "Prediction2": 0}
                for j in range(2):
                    dict_res["Prediction" + str(j + 1)] = predicted[i][j].item()
                df = df.append(dict_res, ignore_index=True)
    df.to_csv(Configure.result_dir, index=0)
 
 
def save_model(target_model, model_path):
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(target_model.state_dict(), model_path)
 
 
def load_model(target_model, model_path):
    if not os.path.exists(model_path):
        print("模型路径错误,模型加载失败")
        sys.exit(0)
    else:
        target_model.load_state_dict(torch.load(model_path))
        target_model.eval()
 
 
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
    Configure = Configure()

    train_dataset = MalwareDataset(Configure.train_path)
    train_loader = DataLoader(train_dataset, batch_size=Configure.batch_size,
                              shuffle=True, num_workers=2)


    test_dataset = MalwareTestDataset(Configure.test_path)
    test_loader = DataLoader(test_dataset, batch_size=Configure.batch_size,
                             shuffle=False, num_workers=2)
 
    if Configure.load_model:    # 选择加载模型
        model = vgg16()
        print("=====================开始加载模型================")
        load_model(model, Configure.model_path)
        print("=====================模型加载完成================")
    else:   # 选择训练模型
        model = vgg16(pretrained=True)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    if not Configure.load_model:
        optimizer = torch.optim.SGD(model.parameters(), lr=Configure.lr,
                                    weight_decay=Configure.decay, momentum=Configure.momentum)    # 定义优化器
        print("=====================开始训练模型================")
        for i in range(Configure.epochs):
            train(i)
        print("=====================模型训练完成================")
        save_model(model, Configure.model_path)
    print("=====================开始测试模型================")
    test()
    print("=====================模型测试完成================")