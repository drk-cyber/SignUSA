from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results=model.train(
        data='C:/Users/24331/Desktop/SignUSA/data.yaml',
        epochs=50,           # 把全部数据学 50 遍
        batch=16,            # 每批 16 张图一起算
        imgsz=416,           # 输入图片尺寸，和数据集预处理一致
        name='sign_lesson',  # 实验名，结果会存到 runs/detect/sign_lesson/
        patience=10,         # 连续 10 轮没进步就自动停止
        device=0,           # 用第 0 号 GPU

    )