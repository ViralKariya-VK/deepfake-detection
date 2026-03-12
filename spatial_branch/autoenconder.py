import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class FaceDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.filenames  = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(".jpg") or f.endswith(".jpeg")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path  = os.path.join(self.images_dir, self.filenames[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderTrainer:
    def __init__(self, images_dir, checkpoint_dir, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.dataset    = FaceDataset(images_dir, transform=self.transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        self.model     = ConvAutoencoder().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train(self, epochs=20):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch         = batch.to(self.device)
                reconstructed = self.model(batch)
                loss          = self.criterion(reconstructed, batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")

            if (epoch + 1) % 5 == 0:
                self.save(f"autoencoder_epoch{epoch+1}.pth")

        self.save("autoencoder_final.pth")

    def get_reconstruction_error(self, image_path):
        self.model.eval()
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(tensor)

        error   = torch.abs(tensor - reconstructed)
        heatmap = error.squeeze(0).mean(dim=0).cpu().numpy()
        score   = float(heatmap.mean())

        return score, heatmap

    def save(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Saved: {path}")

    def load(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Loaded: {path}")