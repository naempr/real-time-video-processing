from ultralytics import YOLO

class DetectorTracker:
    def __init__(self, model_path, tracker_yaml, conf=0.3, iou=0.5, imgsz=960, classes=None):
        self.model = YOLO(model_path)
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes if classes is not None else [0]

    def infer(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False
        )
        return results[0]
