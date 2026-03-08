# preprocessing/videos_to_frame.py

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessing.face_detector import FaceDetector
from preprocessing.face_aligner import FaceAligner

# tqdm gives you a progress bar — essential when processing 160k images.

class DatasetProcessor:
    def __init__(self, celeba_root, output_dir, output_size=224):
        self.celeba_root = celeba_root
        self.images_dir = os.path.join(celeba_root, 'img_align_celeba', 'img_align_celeba')
        self.output_dir = output_dir
        self.detector = FaceDetector(min_confidence=0.9)
        self.aligner = FaceAligner(output_size=output_size)

        if not os.access(os.path.dirname(output_dir) or '/', os.W_OK):
            raise OSError(f"Cannot write to {output_dir}: read-only or permission denied")
        os.makedirs(output_dir, exist_ok=True)

    # We initialize the detector and aligner once here — not inside the loop. This is important for performance.

    def load_celeba_metadata(self):
        partition = pd.read_csv(
            os.path.join(self.celeba_root, 'list_eval_partition.csv')
        )
        bbox = pd.read_csv(
            os.path.join(self.celeba_root, 'list_bbox_celeba.csv')
        )
        landmarks = pd.read_csv(
            os.path.join(self.celeba_root, 'list_landmarks_align_celeba.csv')
        )

        # Only keep training split(partition == 0)
        train_images = partition[partition['partition'] == 0]['image_id'].tolist()

        return train_images, bbox, landmarks
    

    def get_fallback_landmarks(self, image_id, landmarks_df):
        row = landmarks_df[landmarks_df['image_id'] == image_id].iloc[0]

        return {
            "left_eye": (row['lefteye_x'], row['lefteye_y']),
            "right_eye": (row['righteye_x'], row['righteye_y']),
            "nose": (row['nose_x'], row['nose_y']),
        }
    
    def get_fallback_bbox(self, imag_id, bbox_df):
        row = bbox_df[bbox_df['image_id'] == imag_id].iloc[0]
        
        # CelebA bbox is stored as x, y, width, height — we convert to x1, y1, x2, y2 format because that's what our aligner expects.
        x, y, w, h = row['x1'], row['y1'], row['width'], row['height']
        return [x, y, x + w, y + h]
    

    def process_celeba(self, limit=None):
        train_images, bbox_df, landmarks_df = self.load_celeba_metadata()

        if limit:
            train_images = train_images[:limit]

        success, failed = 0, 0

        for image_id in tqdm(train_images, desc="Processing CelebA Dataset"):
            image_path = os.path.join(self.images_dir, image_id)
            output_path = os.path.join(self.output_dir, image_id)

            # Skip if already processed (allows resuming)
            if os.path.exists(output_path):
                continue

            try:
                image = self.detector.load_image(image_path)
                results = self.detector.detect(image)

                if results: 
                    # RetinaFace successed - use its landmarks
                    face = results[0]  
                    aligned = self.aligner.align(image, face['landmarks'], face['box'])
                else:
                    landmarks = self.get_fallback_landmarks(image_id, landmarks_df)
                    box = self.get_fallback_box(image_id, bbox_df)
                    aligned = self.aligner.align(image, landmarks, box)

                cv2.imwrite(output_path, cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
                success += 1
 
            except Exception as e:
                failed += 1
                continue

        print(f"\nDone. Succes: {success} | Failed: {failed}")