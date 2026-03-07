import cv2
import numpy as np
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self, min_confidence=0.9):
        self.min_confidence = min_confidence

    # OpenCV reads images in BGR format by default (historical reason). We convert to RGB because every other library (PyTorch, PIL, RetinaFace) expects RGB.

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
        
    def detect(self, image):
        faces = RetinaFace.detect_faces(image)
        results = []

        # RetinaFace returns a dictionary when faces are found, and a non-dict value when nothing is detected. So we check for that before proceeding.
        if not isinstance(faces, dict):
            return results                  # No faces found, return empty list
        
        # For each detected face, we check if confidence meets our threshold. If yes, we extract the bounding box and landmarks and store them in a clean dictionary.
        for key, face_data in faces.items():
            confidence = face_data['score']

            if confidence < self.min_confidence:
                continue  # Skip faces that don't meet the confidence threshold

            box = face_data['facial_area']  # [x1, y1, x2, y2]
            landmarks = face_data['landmarks']  # {'left_eye': [x, y], 'right_eye': [x, y], 'nose': [x, y], 'mouth_left': [x, y], 'mouth_right': [x, y]}

            results.append({
                "box": box,
                "landmarks": landmarks,
                "confidence": confidence
            })

            return results
        

        