# This file is to train the autoencoder on the embeddings we extracted from the CelebA dataset.

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from spatial_branch.autoenconder import AutoencoderTrainer

trainer = AutoencoderTrainer(
    images_dir     = "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/processed/celeba_aligned",
    checkpoint_dir = "checkpoints/"
)

trainer.train(epochs=20)