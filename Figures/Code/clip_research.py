import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from clip import clip

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Define the image transformation
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

path = "SyncDreamer/my_testset/"

# Load the images
pig1 = Image.open(path +"pig1.png").convert('RGB')
pig2 = Image.open(path +"pig2.png").convert('RGB')
bunny1 = Image.open(path +"bunny1.png").convert('RGB')
bunny2 = Image.open(path +"bunny2.png").convert('RGB')

# Preprocess the images
pig1 = transform(pig1).unsqueeze(0).to(device)
pig2 = transform(pig2).unsqueeze(0).to(device)
bunny1 = transform(bunny1).unsqueeze(0).to(device)
bunny2 = transform(bunny2).unsqueeze(0).to(device)

# Embed the images
with torch.no_grad():
    pig1_embed = model.encode_image(pig1)
    pig2_embed = model.encode_image(pig2)
    bunny1_embed = model.encode_image(bunny1)
    bunny2_embed = model.encode_image(bunny2)

cosi = torch.nn.CosineSimilarity(dim=0)
print(cosi(pig1_embed[0,:], pig2_embed[0,:]))
print(cosi(pig1_embed[0,:], bunny1_embed[0,:]))
print(cosi(pig1_embed[0,:], bunny2_embed[0,:]))
print(cosi(pig2_embed[0,:], bunny1_embed[0,:]))
print(cosi(pig2_embed[0,:], bunny2_embed[0,:]))
print(cosi(bunny1_embed[0,:], bunny2_embed[0,:]))

# Output:
# tensor(0.9321, device='cuda:0', dtype=torch.float16)
# tensor(0.7744, device='cuda:0', dtype=torch.float16)
# tensor(0.7783, device='cuda:0', dtype=torch.float16)
# tensor(0.7729, device='cuda:0', dtype=torch.float16)
# tensor(0.7891, device='cuda:0', dtype=torch.float16)
# tensor(0.9644, device='cuda:0', dtype=torch.float16)