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

# Load the images
img1 = Image.open("img2.png")
img2 = Image.open("img1.png")

# Preprocess the images
img1 = transform(img1).unsqueeze(0).to(device)
img2_original = transform(img2).clone().unsqueeze(0).to(device)
img2 = img2_original.clone()

# Embed the images
with torch.no_grad():
    img1_embed = model.encode_image(img1)
    img2_embed = model.encode_image(img2)

# run the clip optimization
optimizer = torch.optim.Adam([img2.requires_grad_()], lr=0.1)
for i in range(3):
    optimizer.zero_grad()
    img2_embed = model.encode_image(img2)
    loss = -torch.cosine_similarity(img1_embed, img2_embed).mean()
    loss.backward()
    optimizer.step()

# reverse the transform
inverse_transform = Compose([
    Normalize(
        mean=(-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711),
        std=(1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711)
    )
])
img1 = inverse_transform(img1).squeeze().permute(1, 2, 0).cpu().detach().numpy()
img2_original = inverse_transform(img2_original).squeeze().permute(1, 2, 0).cpu().detach().numpy()
img2 = inverse_transform(img2).squeeze().permute(1, 2, 0).cpu().detach().numpy()

# display images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title('Image 2')
plt.subplot(1, 3, 2)
plt.imshow(img2_original)
plt.title('Image 1')
plt.subplot(1, 3, 3)
plt.imshow(img2)
plt.title('Modified Image 1')
plt.show()