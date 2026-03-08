# This file is to visualize the t-SNE plot of the embeddings we extracted from the CelebA dataset.
# We will use the visualize_embeddings function from evaluation/visualizer.py to do this.
# The function will load the embeddings, run t-SNE, and save a scatter plot of the 2D t-SNE projections. 
# This helps us see if the embeddings form meaningful clusters or if they are just scattered randomly.

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from evaluation.visualizer import visualize_embeddings

visualize_embeddings(
    embeddings_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/embeddings",
    output_dir="/Users/viral/NMIMS/2. Sem-II/2. UML/Project/project-DeepFake/data/"
)


'''
What to Look For in the Output

**Good sign:**
```
Points form one or a few tight clusters
→ DINO is producing consistent representations
```

**Bad sign:**
```
Points completely scattered with no structure
→ something wrong in preprocessing or embedding extraction'''