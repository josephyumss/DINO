import torch
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
model.eval()

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0) # insert batch dim (n,m) -> (1,n,m) for model inference
    with torch.no_grad(): # use model without gradient update, only for inference
        feature_dict= model.forward_features(img) # shape : (1, feature_dim)
        feature = feature_dict['x_norm_clstoken']
    return feature.squeeze(0).cpu().numpy() # shape : (feature_dim,)

fire_imgs = r"C:\Users\User\Desktop\랩실\wild_fire_dataset\archive\the_wildfire_dataset_2n_version\train\fire"
nofire_imgs = r"C:\Users\User\Desktop\랩실\wild_fire_dataset\archive\the_wildfire_dataset_2n_version\train\nofire"

x = []
label = []

for fire_img in os.listdir(fire_imgs):
    path = os.path.join(fire_imgs, fire_img)
    x.append(extract_feature(path))
    label.append(1)

for nofire_img in os.listdir(nofire_imgs):
    path = os.path.join(nofire_imgs, nofire_img)
    x.append(extract_feature(path))
    label.append(0)

x = np.array(x)
label = np.array(label)

print(f"feature shape : {x.shape}, Labels : {label.shape}")
print(f"x max : {np.max(x)}, x min : {np.min(x)}")

X_train, X_test, y_train, y_test = train_test_split(x,label,test_size=0.2,random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)
confusion_mat = confusion_matrix(y_test,y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Recall : {recall:.4f}')
print('Confusion Matrix:')
print(confusion_mat)