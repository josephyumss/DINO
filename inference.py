import torch
from train import extract_feature, LogisticRegression
import torchvision.transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

img_path = r"sample_data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

time_tic_load = time.time()
DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
time_toc_load = time.time()
DINO.eval()

hour = int((time_toc_load-time_tic_load)//3600)
mint = int(((time_toc_load-time_tic_load)%3600)//60)
sec = (time_toc_load-time_tic_load)%60 
print(f"Loading DINO : {hour}:{mint}:{format(sec, '4.2f')} ({(time_toc_load-time_tic_load)*1000:.2f} ms) Done")

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

x=[]
images = []

time_tic_DINO = time.time()
for img in os.listdir(img_path):
    path = os.path.join(img_path, img)
    img = Image.open(path).convert('RGB') 
    img = np.array(img)
    x.append(extract_feature(path, DINO, device))
    images.append(img)
time_toc_DINO = time.time()

hour = int((time_toc_DINO-time_tic_DINO)//3600)
mint = int(((time_toc_DINO-time_tic_DINO)%3600)//60)
sec = (time_toc_DINO-time_tic_DINO)%60 
print(f"DINO extracting features : {hour}:{mint}:{format(sec, '4.2f')} ({(time_toc_DINO-time_tic_DINO)*1000:.2f} ms) Done")

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

time_tic_log = time.time()
pred = classifier(x).detach().numpy()
pred_class = (pred>0.5).astype(int)
time_toc_log = time.time()

hour = int((time_toc_log-time_tic_log)//3600)
mint = int(((time_toc_log-time_tic_log)%3600)//60)
sec = (time_toc_log-time_tic_log)%60 
print(f"Logistic Regrssion Classification : {hour}:{mint}:{format(sec, '4.2f')} ({(time_toc_log-time_tic_log)*1000:.2f} ms) Done")

hour = int((time_toc_log-time_tic_load)//3600)
mint = int(((time_toc_log-time_tic_load)%3600)//60)
sec = (time_toc_log-time_tic_load)%60 
print("------------------------------------")
print(f"Total Inference : {hour}:{mint}:{format(sec, '4.2f')} ({(time_toc_log-time_tic_load)*1000:.2f} ms) Done")
print("------------------------------------")

fig, axes = plt.subplots(2,5, figsize=(20,10))
idx = 0
for i in range(2):
    for j in range(5):
        sample_img = cv2.resize(images[idx], (200,200))
        sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
        axes[i][j].imshow(sample_img)
        axes[i][j].axis('off')
        axes[i][j].set_title(f"label : {label[idx]}\n pred : {pred_class[idx]}", fontsize=10)
        idx += 1
plt.tight_layout()
plt.show()
