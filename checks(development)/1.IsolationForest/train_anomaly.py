# This file is to train the Isolation Forest anomaly detector on the embeddings 
# we extracted from the CelebA dataset.
# This provides us the distribution of real face embeddings, 
# which we can then use to score new images (real or fake) 
# and see how anomalous they are compared to the real distribution.


import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
from spatial_branch.anomaly import AnomalyDetector

# Load your saved embeddings
embeddings = np.load("/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/embeddings/embeddings.npy")
print(f"Embeddings shape: {embeddings.shape}")  # (10000, 384)

# Fit on real faces
detector = AnomalyDetector()
detector.fit(embeddings)

# Score the same embeddings (sanity check)
scores = detector.score(embeddings)
print(f"Score range: {scores.min():.3f} to {scores.max():.3f}")
print(f"Mean score:  {scores.mean():.3f}")

# Save the model
detector.save("checkpoints/isolation_forest.pkl")
