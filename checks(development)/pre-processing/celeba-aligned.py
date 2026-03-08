# app/processing_celeba_df.py

# This python file is to preprocess the celeba dataset and save the aligned and resized images in a new directory. 
# We will use the DatasetProcessor class from preprocessing/videos_to_frame.py to do this.
# The process_celeba function will read the original images, align the faces using MTCNN, resize them to 224x224, and save them in the output directory. 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.vidoes_to_frame import DatasetProcessor

processor = DatasetProcessor(
    celeba_root="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive",
    output_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/processed/celeba_aligned",
    output_size=224
)

# Test on X images first
processor.process_celeba(limit=None)