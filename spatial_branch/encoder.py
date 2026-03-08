# spatial_branch/encoder.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

## we import transforms from torchvision. This is how we convert our face images into the exact format DINO expects.

'''
Why these specific mean/std values?
These are the ImageNet mean and standard deviation — the dataset DINO was trained on. Normalizing with these values puts pixel values in the same range the model saw during training.
Think of it like this — if the model learned with data scaled a certain way, you must scale new data the same way.
'''

class DINOEncoder:
    def __init__(self, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps") # M4 Mac
            elif torch.cuda.is_available():
                self.device = torch.device("cuda") # Colab GPU
            else:
                self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # ---Loading DINO Model---------------
        self.model = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits8'
            )
        self.model.eval()
        self.model.to(self.device)

        '''What is dino_vits8
            dino  →  the training method
            vit   →  Vision Transformer architecture
            s     →  "small" variant (good balance of speed vs quality)
            8     →  patch size of 8×8 pixels
        '''

        # DINO was trained with specific preprocessing. If we feed images differently, the embeddings will be garbage. The required transforms are:
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    # ---Single Image Embeddings--------------- 
    def get_embedding(self, image):
        # image can be numpy array (from OpenCV) or PIL Image

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, np.ndarray):
            image = Image.formarray(image)

        tensor = self.transform(image)                      # shape: [3, 224, 224]
        tensor = tensor.unsqueeze(0).to(self.device)        # shape: [1, 3, 224, 224]

        with torch.no_grad():
            embedding = self.model(tensor)                  # shape: [1, 384]               

        embedding = self.model(tensor)
        embedding = embedding.squeeze().detach().cpu().numpy()
        return embedding
    
        '''
        Why unsqueeze(0)?
        DINO expects a batch of images — shape [batch, channels, height, width]. We have one image so we add a batch dimension of 1. After getting the result we squeeze(0) to remove it.
        Why torch.no_grad()?
        We're not training — just running forward pass. no_grad() tells PyTorch not to store gradients, saving memory and making it faster.
        '''

    
    # ---Batch Processing---------------
    def get_embeddings_batch(self, image_paths, batch_size=32):
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i: i + batch_size]
            batch_tensors = []

            for path in batch_paths: 
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model(batch)
            
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
    

       # Processing images one by one is slow. Batching sends 32 images through the model simultaneously — much faster. batch_size=32 works fine on M4, increase to 64 on Colab.
        