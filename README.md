# 手语字母检测 — 从零到实时识别

## 一、项目概览

| 项目 | 说明 |
|------|------|
| 数据集 | American Sign Language Letters（美国手语字母） |
| 任务 | 目标检测（Object Detection），识别 26 个手语字母 |
| 图片数 | 1728 张（训练 1512 / 验证 144 / 测试 72） |
| 图片尺寸 | 416×416 |
| 类别 | A–Z 共 26 类 |
| 标注格式 | YOLO 格式 |
| 来源 | Roboflow Universe |

## 二、目录结构

```
SignUSA/
├── data.yaml              ← 数据集配置文件（告诉模型数据在哪、有哪些类）
├── train/
│   ├── images/   (1512张)
│   └── labels/   (YOLO 格式标注)
├── valid/
│   ├── images/   (144张)
│   └── labels/
├── test/
│   ├── images/   (72张)
│   └── labels/
├── train.py               ← 训练脚本
├── predict.py             ← 图片推理脚本
├── webcam.py              ← 实时摄像头脚本
└── runs/detect/           ← 训练结果
```

## 三、环境搭建

### 需要的工具

| 工具 | 作用 |
|------|------|
| conda | 管理 Python 虚拟环境，隔离不同项目的依赖 |
| PyTorch + CUDA | 深度学习框架 + GPU 加速 |
| ultralytics | YOLO 模型的训练/推理/导出工具包 |

### 关键命令

```bash
conda activate yolov8       # 激活环境
pip install ultralytics     # 安装 YOLO 工具包
pip list                    # 查看已安装的包
```

## 四、数据集配置：data.yaml

```yaml
train: C:\Users\24331\Desktop\SignUSA\train\images
val: C:\Users\24331\Desktop\SignUSA\valid\images
test: C:\Users\24331\Desktop\SignUSA\test\images

nc: 26
names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
```

- `nc`：number of classes，类别数量
- `names`：每个类别的名称，按 ID 顺序排列（0=A, 1=B, ..., 25=Z）
- 路径建议用绝对路径，避免运行目录不同导致找不到数据

## 五、YOLO 标注格式

```
<class_id> <x_center> <y_center> <width> <height>
```

| 字段 | 含义 | 示例 |
|------|------|------|
| class_id | 类别编号 | 0=A, 1=B, ... |
| x_center | 框中心 x 坐标（归一化 0~1） | 0.506 |
| y_center | 框中心 y 坐标（归一化 0~1） | 0.484 |
| width | 框宽度（归一化 0~1） | 0.601 |
| height | 框高度（归一化 0~1） | 0.478 |

所有坐标都是相对值（除以图片宽高），与图片实际尺寸无关。

## 六、训练脚本：train.py

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")    # 加载预训练模型

    results = model.train(
        data="C:/Users/24331/Desktop/SignUSA/data.yaml",
        epochs=50,        # 把数据反复学 50 遍
        imgsz=416,        # 输入图片尺寸
        batch=16,         # 每批处理 16 张
        name="sign_lesson",  # 实验名
        patience=10,      # 连续 10 轮没进步就自动停止
        device=0,         # 用第 0 号 GPU（CPU 用 'cpu'）
    )
```

### 参数解释

| 参数 | 含义 | 调参思路 |
|------|------|----------|
| epochs | 数据集完整过多少遍 | 太少欠拟合，太多过拟合，50~100 通常合适 |
| imgsz | 输入图片尺寸 | 和数据集预处理保持一致 |
| batch | 每批几张图 | 显存不够就调小，一般是 8/16/32 |
| patience | 早停耐心值 | 设 0 关闭早停，一直跑到 epochs |
| device | 用哪个 GPU | 多卡可设 `[0,1]`，CPU 设 `'cpu'` |

### ⚠️ Windows 必须注意

```python
if __name__ == '__main__':
    # 所有训练/推理代码放这里
```

不加这个会导致多进程加载数据时报错：`RuntimeError: An attempt has been made to start a new process...`

原因：Windows 创建子进程时会重新导入你的脚本，不加保护会导致递归调用。

### YOLO 模型尺寸选择

| 模型 | 大小 | 适用场景 |
|------|------|----------|
| yolov8n | 6 MB | 快速实验、轻量设备 |
| yolov8s | 22 MB | 精度稍高 |
| yolov8m | 52 MB | 常规选择 |
| yolov8l | 90 MB | 精度优先 |
| yolov8x | 130 MB | 最高精度 |

## 七、看懂训练日志

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  22/50      1.13G     0.615     0.897      1.022         36        416
```

| 指标 | 含义 | 趋势 |
|------|------|------|
| Epoch | 当前轮次/总轮次 | — |
| GPU_mem | 显存占用 | 稳定就好 |
| box_loss | 边界框位置误差 | 越小越好 ↓ |
| cls_loss | 分类误差（把 A 认成 B） | 越小越好 ↓ |
| dfl_loss | 框形状回归误差 | 越小越好 ↓ |

### 验证结果

```
Class   Images   Instances   Box(P)   R       mAP50   mAP50-95
all     144      144         0.914    0.884   0.947   0.778
```

| 指标 | 全称 | 含义 |
|------|------|------|
| P (Precision) | 精确率 | 模型说是 A 的框里，有多少真的是 A |
| R (Recall) | 召回率 | 所有真的 A 里，模型找到了多少 |
| mAP50 | 平均精度（IoU≥0.5） | 综合指标，最常用，越高越好 |
| mAP50-95 | 平均精度（IoU 0.5~0.95） | 更严格，框要更准 |

### EarlyStopping（早停）

如果连续 `patience` 轮没进步，训练自动停止，保存最佳模型为 `best.pt`。

```
EarlyStopping: Training stopped early as no improvement observed in last 10 epochs.
Best results observed at epoch 22, best model saved as best.pt.
```

## 八、推理脚本：predict.py

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("路径/best.pt")    # 加载自己训练的模型

    results = model.predict(
        source="图片或文件夹路径",
        save=True,         # 保存画了框的结果图
        conf=0.5,          # 置信度阈值（<0.5 的框不显示）
        project="保存目录",
        name="子目录名",
    )
```

## 九、实时摄像头：webcam.py

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("路径/best.pt")

    results = model.predict(
        source=0,      # 0=自带摄像头
        show=True,     # 弹窗显示
        conf=0.5,
        stream=True,   # 流式模式
    )

    for r in results:   # ⚠️ 必须有这个循环，否则不处理帧
        pass
```

按 `q` 键退出。

## 十、完整流程速查

```
① 准备数据集（图片 + YOLO 格式标注）
② 配置 data.yaml（路径、类别数、类别名）
③ conda activate 环境
④ pip install ultralytics（如果没装）
⑤ 写 train.py（加载预训练模型 → model.train()）
⑥ python train.py
⑦ 看训练日志，确认 loss 下降、mAP 合理
⑧ 写 predict.py / webcam.py 用 best.pt 做推理
```

## 十一、常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| 无法解析导入 "ultralytics" | VSCode 解释器没选对 | Ctrl+Shift+P → Python: Select Interpreter → 选 yolov8 环境 |
| RuntimeError: multiprocessing... | Windows 多进程问题 | 加 `if __name__ == '__main__':` |
| grep 不是内部命令 | Windows 没有 grep | 用 `findstr` 替代，或装 Git Bash |
| 训练自动提前停止 | patience 触发早停 | 正常现象，最佳模型已保存。如需继续跑，设 `patience=0` |
| stream=True 没反应 | 没写 for 循环 | 必须 `for r in results: pass` |
