import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_branch.encoder import DINOEncoder

IMAGE_PATH = 'output/celeba_aligned/000001.jpg'

encoder = DINOEncoder()

# Test single image
embedding = encoder.get_embedding(IMAGE_PATH)
print(f"Embedding shape: {embedding.shape}")   # Should be (384,)
print(f"Embedding sample: {embedding[:5]}")    # First 5 values