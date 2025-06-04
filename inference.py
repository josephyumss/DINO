import torch
from train import extract_feature, LogisticRegression
import torchvision.transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img_path = r"sample_data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
DINO.eval()

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

x=[]
images = []
for img in os.listdir(img_path):
    path = os.path.join(img_path, img)
    img = Image.open(path).convert('RGB') 
    img = np.array(img)
    x.append(extract_feature(path, DINO, device))
    images.append(img)

x = torch.tensor(x, dtype=torch.float32)
print(x.shape)
label = [1,1,1,1,1,0,0,0,0,0]

classifier = LogisticRegression(x.shape[-1])

if device == 'cpu':
    weights = torch.load('classifier_weights.pth',map_location=torch.device('cpu'))
    classifier.load_state_dict(weights)
else :
    classifier.load_state_dict(torch.load('classifier_weights.pth'))

classifier.eval()

pred = classifier(x).detach().numpy()
pred_class = (pred>0.5).astype(int)

fig, axes = plt.subplots(2,5)
idx = 0
for i in range(2):
    for j in range(5):
        sample_img = cv2.resize(images[idx], (30,30))
        sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
        axes[i][j].imshow(sample_img)
        axes[i][j].axis('off')
        axes[i][j].set_title(f"label : {label[idx]}\n pred : {pred_class[idx]}", fontsize=10)
        idx += 1
plt.tight_layout()
plt.show()
