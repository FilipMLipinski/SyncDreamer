import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from clip import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device)

# Define the image transformation
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# Load the images
url1 = "https://placekitten.com/800/600"
url2 = "https://placekitten.com/800/601"
response1 = requests.get(url1)
response2 = requests.get(url2)
img1 = Image.open(BytesIO(response1.content))
img2 = Image.open(BytesIO(response2.content))

# Preprocess the images
img1 = transform(img1).unsqueeze(0).to(device)
img2_original = transform(img2).clone().unsqueeze(0).to(device)
img2 = img2_original.clone()

# Embed the images
with torch.no_grad():
    img1_embed = model.encode_image(img1)
    img2_embed = model.encode_image(img2)

#Make img2 more like img1
optimizer = torch.optim.Adam([img2.requires_grad_()], lr=0.1)
for i in range(10):
    optimizer.zero_grad()
    img2_embed = model.encode_image(img2)
    loss = -torch.cosine_similarity(img1_embed, img2_embed).mean()
    loss.backward()
    optimizer.step()
    if(i%100 == 0):
      print(loss)

# Display the images
img1 = img1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
img2_original = img2_original.squeeze().permute(1, 2, 0).cpu().detach().numpy()
img2 = img2.squeeze().permute(1, 2, 0).cpu().detach().numpy()

orig_1 = Image.fromarray(img1)
orig_2 = Image.fromarray(img2_original)
new_2 = Image.fromarray(img2)
new_2.save("new_2.png")