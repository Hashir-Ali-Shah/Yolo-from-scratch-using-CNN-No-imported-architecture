from ultralytics import YOLO
from utils import letterbox,read_image

model_path = r"D:\ML\Tasks\CNN\Yolo_results\weapons_train2\weights\last.pt"
image_path = r"D:\ML\Tasks\CNN\All-weapons-data-1\test\images\test_0002.jpg"

model=YOLO(model_path)
results=model.predict(source=image_path,conf=0.25,save=True,save_txt=True,save_conf=True)
