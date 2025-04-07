from ultralytics import YOLO

import torch

# 释放未使用的 GPU 内存
torch.cuda.empty_cache()
# 检查是否有可用的 GPU
# 创建一个yolo类的对象, 调用YOLO的构造函数
model = YOLO("/home/sys422/qfy/project/ultralytics/yaml2/c2f-f.yaml", task="detect")# 加载模型

# 使用模型
model.train(data="/home/sys422/qfy/project/ultralytics/VisDrone.yaml",
            epochs=150,
            batch=16,
            imgsz=640,
            device = [2],
            project ='VisDrone',
            name = "VisDrone_c2f-f"
            )  # 训练模型