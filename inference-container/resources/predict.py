import os
import numpy as np
import json
import base64
from ultralytics import YOLO
from ultralytics.tracker.track import register_tracker
from flask import Flask, Response, request

app = Flask(__name__)

class YOLOService:
    model = None
    model_name = "/opt/ml/model/yolov8n-pose.pt"
    tracking = True
    classes = [0]
    conf = 0.25
    iou = 0.7
    half = False
    def __init__(self):
        self.model_name = f"/opt/ml/model/{os.environ.get('SM_MODEL_NAME', 'yolov8n-pose.pt')}"
        self.tracking = os.environ.get('SM_TRACKING', 'Enabled')=='Enabled'
        self.classes = json.loads(os.environ.get('SM_HP_CLASSES', '[0]'))
        self.conf = float(os.environ.get('SM_HP_CONF', '0.25'))
        self.iou = float(os.environ.get('SM_HP_IOU', '0.7'))
        self.half = os.environ.get('SM_HP_HALF', 'False')=='True'
        print("YOLO configuration")
        print("======================")
        print(f"Model Name: {self.model_name}")
        print(f"Tracking: {'Enabled' if self.tracking else 'Disabled'}")
        print(f"Classes: {self.classes}")
        print(f"Confidence: {self.conf}")
        print(f"Intersection over union (iou): {self.iou}")
        print(f"Half percision: {'Enabled' if self.half else 'Disabled'}")
    def load_model(self):
        print(f"Loading YoloV8 model from {self.model_name}")
        self.model = YOLO(self.model_name)
        if YOLO_class.tracking:
            register_tracker(YOLO_class.model, persist=True)
    
def tojson(Results, normalize=False):
    """Convert the object to JSON format."""

    # Create list of detection dictionaries
    results = []
    data = Results.boxes.data.cpu().tolist()
    h, w = Results.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):
        box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
        track_id = row[4]
        conf = row[5]
        id = int(row[6])
        name = Results.names[id]
        result = {'name': name, 'class': id, 'confidence': conf, 'box': box, 'id': track_id}
        if Results.keypoints is not None:
            x, y, visible = Results.keypoints[i].cpu().unbind(dim=1)  # torch Tensor
            result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
        results.append(result)
    
    # Convert detections to JSON
    return results

@app.route("/ping", methods=["GET"])
def health_check():
    """Determine if the container is working and healthy. In this container, we declare
    it healthy if we can load the model successfully and create a predictor."""
    healthy = True if YOLO_class.model else False
    status = 200 if healthy else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def inference():
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        return Response(response=result, status=415, mimetype="application/json")

    try:
        content = request.get_json()
        frame_id = content["frame_id"]
        
        print(f'Received Frame-{frame_id}')
        if int(frame_id) == 0:
            YOLO_class.load_model()
                    
        decoded_bytes = base64.decodebytes(content["frame_data"].encode("utf-8"))
        img_array = np.frombuffer(decoded_bytes, dtype=np.uint8).reshape((content["frame_h"],content["frame_w"], 3))
        results = YOLO_class.model.predict(
            img_array, 
            conf=YOLO_class.conf, 
            iou=YOLO_class.iou, 
            half=YOLO_class.half, 
            classes=YOLO_class.classes
        )
        response = [tojson(result) for result in results]
        return Response(response=json.dumps(response), status=200, mimetype="application/json")
    except Exception as e:
        print(str(e))
        result = {"error": f"Internal server error"}
        return Response(response=result, status=500, mimetype="application/json")
    
YOLO_class = YOLOService()
YOLO_class.load_model()