from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import torch
import numpy as np
from torchvision.utils import save_image
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
from collections import OrderedDict

def create_latent_vector(prompt, latent_size=128):
    # Convert prompt to a numpy array, use a fixed seed for reproducibility
    np.random.seed(int(prompt))
    latent_vector = np.random.normal(0, 1, latent_size)

    # Convert to torch tensor
    latent_tensor = torch.tensor(latent_vector, dtype=torch.float32).reshape(1, latent_size, 1, 1)
    return latent_tensor

app = FastAPI()

# Load the generator model (assumes it's defined and weights are loaded)
# Ensure you load your trained generator model here

# Your Generator Model Definition here
class Generator(nn.Module):
    def __init__(self, latent_size, image_size, hidden_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.image_size = image_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 8, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 4, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load the models
latent_size = 128  # Set latent_size to 128
image_size = 64
hidden_size = 64

generator = Generator(latent_size, image_size, hidden_size)
discriminator = Discriminator(image_size, hidden_size)

# Load the saved state dictionaries, mapping to CPU
state_dict_G = torch.load('G.ckpt', map_location=torch.device('cpu'))

# Modify the keys in the state_dict for Generator
new_state_dict_G = OrderedDict()
for k, v in state_dict_G.items():
    new_state_dict_G['net.' + k] = v

# Load the modified state dictionary for Generator
generator.load_state_dict(new_state_dict_G)

# Load the saved state dictionaries, mapping to CPU
state_dict_D = torch.load('D.ckpt', map_location=torch.device('cpu'))

# Modify the keys in the state_dict for Discriminator
new_state_dict_D = OrderedDict()
for k, v in state_dict_D.items():
    new_state_dict_D['net.' + k] = v

# Load the modified state dictionary for Discriminator
discriminator.load_state_dict(new_state_dict_D)

# Set the models to evaluation mode
generator.eval()
discriminator.eval()

# Define transformations
image_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)])

@app.get("/generate/{prompt}")
async def generate_image(prompt: int):
    try:
        # Create latent vector from prompt
        latent_vector = create_latent_vector(prompt)

        # Generate image
        with torch.no_grad():
            generated_image = generator(latent_vector)

        # Denormalize image
        generated_image = generated_image * 0.5 + 0.5

        # Save the image to a BytesIO object
        buf = BytesIO()
        save_image(generated_image, buf, format='PNG')
        buf.seek(0)

        # Return the image
        return FileResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
