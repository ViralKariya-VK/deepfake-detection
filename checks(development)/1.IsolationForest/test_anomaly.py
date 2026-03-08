# This file is to test the Isolation Forest anomaly detector on a mix of real and fake images. 
# We will print out the anomaly scores and see if real images get low scores (close to 0) 
# and fake images get high scores (close to 1). 
# This is a sanity check to ensure our model is working as expected before we run it on larger datasets.

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from spatial_branch.encoder import DINOEncoder
from spatial_branch.anomaly import AnomalyDetector

encoder  = DINOEncoder()
detector = AnomalyDetector()
detector.load("checkpoints/isolation_forest.pkl")

fake_images = [
]

import numpy as np
from spatial_branch.encoder import DINOEncoder
from spatial_branch.anomaly import AnomalyDetector

encoder  = DINOEncoder()
detector = AnomalyDetector()
detector.load("checkpoints/isolation_forest.pkl")

# Mix of real and fake paths
real_paths = [
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000001.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000002.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000003.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000004.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000005.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000006.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000007.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000008.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000009.jpg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/raw/celeba/archive/img_align_celeba/img_align_celeba/000010.jpg",
]

fake_paths = [
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-1.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-2.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-3.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-4.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-5.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-6.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-7.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-8.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-9.jpeg",
    "/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/deepfake_photos_for-testing/image-10.jpeg",
]

all_paths  = real_paths + fake_paths
all_labels = ["REAL"] * len(real_paths) + ["FAKE"] * len(fake_paths)

# Extract all embeddings
all_embeddings = []
for path in all_paths:
    embedding = encoder.get_embedding(path)
    all_embeddings.append(embedding)

all_embeddings = np.array(all_embeddings)

# Score all together — normalization is now across real + fake
scores = detector.score(all_embeddings)

print(f"\n{'Image':<20} {'Label':<8} {'Score':<8} {'Decision'}")
print("─" * 55)
for path, label, score in zip(all_paths, all_labels, scores):
    decision = "FAKE" if score > 0.5 else "REAL"
    flag = "✅" if decision == label else "❌"
    print(f"{path.split('/')[-1]:<20} {label:<8} {score:.3f}    {decision} {flag}")

real_scores = scores[:len(real_paths)]
fake_scores = scores[len(real_paths):]
print(f"\nMean real score: {real_scores.mean():.3f}")
print(f"Mean fake score: {fake_scores.mean():.3f}")
