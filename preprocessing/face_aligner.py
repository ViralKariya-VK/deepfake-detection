# preprocessing/face_aligner.py

import cv2
import numpy as np
from PIL import Image

class FaceAligner:
    def __init__(self, output_size=224):
            # output_size=224 because DINO ViT expects 224×224 images. This is set in config.yaml too so you can change it from one place later.
        self.output_size = output_size

    def align(self, image, landmarks, box):
        # Step 1 - get eye cordinates from landmarks
        left_eye = landmarks["right_eye"]  # Yes, left eye is on the right side of the image and vice versa. This is because the image is in the person's perspective, not ours.
        right_eye = landmarks["left_eye"]

        # Step 2 - calculate the angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))      # arctan2 gives us the angle of the line connecting both eyes relative to horizontal. If the face is perfectly straight, this angle is 0.

        # Step 3 - find center point between eyes
        eye_center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )

        # Step 4 - build rotation matrix and rotate
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        rotated = cv2.warpAffine(image, rotation_matrix,
                                  (image.shape[1], image.shape[0]),
                                  flags=cv2.INTER_LINEAR)                       ## We rotate around the midpoint between the eyes, not the image center. This keeps the face centered during rotation.

        # Step 5 - crop using bounding box
        x1, y1, x2, y2 = box
        
        # Add small padding so we don't cut off chin/forehead
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        cropped = rotated[y1:y2, x1:x2]
        
        # Step 6 - resize to output_size x output_size
        resized = cv2.resize(cropped, (self.output_size, self.output_size))
        
        return resized