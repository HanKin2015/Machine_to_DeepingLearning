import os
 
 
# 存储程序需要的内容
class Configure:
    # 数据集存放路径，根据实际情况更改
    base_path = ""
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "validation")
 
    model_path = "model.pth"    # 保存模型的路径
    load_model = False  # 是否要加载模型
 
    batch_size = 8
    epochs = 25
    lr = 0.001
    decay = 0.0005
    momentum = 0.9