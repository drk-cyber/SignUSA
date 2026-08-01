from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("C:/Users/24331/Desktop/SignUSA/runs/detect/sign_lesson-4/weights/best.pt")

    results = model.predict(
        source=0,
        show=True,
        conf=0.5,
        stream=True,
    )

    for r in results:    # 👈 必须在 for 循环里消费，才会一帧帧处理
        pass
