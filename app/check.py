# app/check.py
# This python file is for running random code

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.face_detector import FaceDetector

IMAGE_PATH = 'app/sample_data/000001.jpg'

detector = FaceDetector()
image = detector.load_image(IMAGE_PATH)
results = detector.detect(image)

print(f"Faces Found: {len(results)}")
for r in results:
    print(r['box'], r['confidence'])
