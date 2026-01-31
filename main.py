from ultralytics import YOLO
from utils import get_device

if __name__ == "__main__":

    device = get_device()

    PRE_TRAINED_MODEL = 'Yolo-Weights/yolov8n.pt'

    model = YOLO(PRE_TRAINED_MODEL)

    results = model.train(data='config.yaml', epochs=100, imgsz=640, device=device)
