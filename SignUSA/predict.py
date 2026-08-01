from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载你训练的模型
    model = YOLO("C:/Users/24331/Desktop/SignUSA/runs/detect/sign_lesson-4/weights/best.pt")
    
    # 2. 对测试集图片做预测
    results = model.predict(
        source="C:/Users/24331/Desktop/SignUSA/test/images",
        save=True,          # 保存结果图片
        conf=0.5,           # 置信度 > 50% 才显示
        project="C:/Users/24331/Desktop/SignUSA/predictions",  # 结果存这儿
        name="lesson_result",
    )
