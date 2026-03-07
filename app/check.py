# app/check.py
# This python file is for running random code

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.vidoes_to_frame import DatasetProcessor

processor = DatasetProcessor(
    celeba_root="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive",
    output_dir="output/celeba_aligned",
    output_size=224
)

# Test on 50 images first
processor.process_celeba(limit=50)