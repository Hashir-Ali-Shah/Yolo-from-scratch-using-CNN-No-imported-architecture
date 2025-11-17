from ultralytics import YOLO
import cv2

class YOLOInference:
    def __init__(self, model_path, device="cuda",GPU=False, imgsz=640, conf=0.25):
        """
        model_path: path to trained YOLO weights
        device: "cuda" or "cpu"
        imgsz: model input size (height & width)
        conf: confidence threshold
        """
        self.model = YOLO(model_path)
        if GPU:
            self.model.to(device)
        self.imgsz = imgsz
        self.conf = conf


    def stream_video_frames(self, video_path):
        """
        Generator that yields annotated frames as NumPy arrays
        """
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
            annotated = results[0].plot()
            yield annotated 

        cap.release()

    def get_full_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
            annotated = results[0].plot()
            frames.append(annotated)

        cap.release()
        return frames


    def process_image(self, image_path=None, frame=None):
        """
        Provide either image_path or frame (numpy array)
        Returns annotated image and raw detection results
        """
        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        annotated = results[0].plot()
        return annotated, results[0]
