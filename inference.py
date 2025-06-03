import torch
from train import extract_feature, LogisticRegression
import torchvision.transforms as T
import numpy as np

img_path = r"sample_data\fire_0.jpg"

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

feature = extract_feature(img_path, DINO, device)
x = torch.tensor(feature.reshape((1,-1)), dtype=torch.float32)
print(x.shape)
label = 1

classifier = LogisticRegression(x.shape[-1])
classifier.load_state_dict(torch.load('classifier_weights.pth'))
classifier.eval()

pred = classifier(x)
print(pred)


