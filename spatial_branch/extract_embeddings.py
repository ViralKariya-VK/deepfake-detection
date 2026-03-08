# spatial_branch/extract_embeddings.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from spatial_branch.encoder import DINOEncoder

def extract_and_save(images_dir, output_dir, batch_size=32):
    # Get all image paths

    filenames = sorted(os.listdir(images_dir))
    '''
    Why sorted()?
    Ensures filenames and embeddings are always in the same order. If order is random, filenames[i] might not match embeddings[i] — that would silently corrupt everything downstream.
    '''

    filenames = [f for f in filenames if f.endswith('.jpg')]
    image_paths = [os.path.join(images_dir, f) for f in filenames]

    print(f"Found {len(image_paths)} images")

    # Load Encoder
    encoder = DINOEncoder()
    
    # Extract embeddings in batches
    all_embeddings = encoder.get_embeddings_batch(
        image_paths,
        batch_size = batch_size
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    np.save(os.path.join(output_dir, "filenames.npy"), np.array(filenames))

    print(f"Saved embeddings shape: {all_embeddings.shape}")
    print(f"Saved to: {output_dir}")


    # if __name__ == "__main__":
    #     extract_and_save(
    #         images_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/processed/celeba_aligned",
    #         output_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/embeddings",
    #         batch_size=32
    #     )

extract_and_save(
    images_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/processed/celeba_aligned",
    output_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/embeddings",
    batch_size=32
)