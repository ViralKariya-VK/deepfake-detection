import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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