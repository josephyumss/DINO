import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os

#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# DINO
def extract_feature(img_path, model, device): # device variable only for read, so global, no need to get from param
        img = Image.open(img_path).convert('RGB') 
        img = transform(img).unsqueeze(0).to(device) # insert batch dim (n,m) -> (1,n,m) for model inference
        with torch.no_grad(): # use model without gradient update, only for inference
            feature_dict= model.forward_features(img) # shape : (1, feature_dim)
            feature = feature_dict['x_norm_clstoken']
        return feature.squeeze(0).cpu().numpy() # shape : (feature_dim,). Because of torch.no_grad(), no need to .detach()

class LogisticRegression(nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.weight = nn.Linear(feature_dim,1)
    
    def forward(self, x):
        return torch.sigmoid(self.weight(x))
    
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device : {device}")

    if not os.path.exists('x_feature.npy'):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
        model.eval()

        #labtop
        # fire_imgs = r"C:\Users\User\Desktop\랩실\wild_fire_dataset\archive\the_wildfire_dataset_2n_version\train\fire"
        # nofire_imgs = r"C:\Users\User\Desktop\랩실\wild_fire_dataset\archive\the_wildfire_dataset_2n_version\train\nofire"

        #desktop
        fire_imgs = r"F:\the_wildfire_dataset_2n_version\train\fire"
        nofire_imgs = r"F:\the_wildfire_dataset_2n_version\train\nofire"

        x = []
        label = []

        for i,fire_img in enumerate(os.listdir(fire_imgs)):
            path = os.path.join(fire_imgs, fire_img)
            x.append(extract_feature(path, model, device))
            label.append(1)
            print(f"feature extract {i}")

        for i,nofire_img in enumerate(os.listdir(nofire_imgs)):
            path = os.path.join(nofire_imgs, nofire_img)
            x.append(extract_feature(path, model, device))
            label.append(0)
            print(f"feature extract {i}")

        x = np.array(x)
        label = np.array(label)

        # save features
        np.save('x_feature.npy', x)
        np.save('label.npy',label)

    else:
        x = np.load('x_feature.npy')
        label = np.load('label.npy')

    print(f"feature shape : {x.shape}, Labels : {label.shape}")
    print(f"x max : {np.max(x)}, x min : {np.min(x)}")

    X_train, X_test, y_train, y_test = train_test_split(x,label,test_size=0.2,random_state=42)

    # to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    classifier = LogisticRegression(x.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)

    epochs = 1000

    for epoch in range(epochs):
        
        classifier.train()
        optimizer.zero_grad()
        train_pred = classifier(X_train)
        loss = criterion(train_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch%10 == 0:
            print(f"epoch : {epoch} | loss : {loss.item():.4f}")
        else :
            print(f"epoch : {epoch} | loss : {loss.item():.4f}")

    # save parameter
    torch.save(classifier.state_dict(), 'classifier_weights.pth')

    # Evaluate
    classifier.eval()
    y_pred = classifier(X_test)

    # for model evaluate
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test,y_pred_class)
    confusion_mat = confusion_matrix(y_test,y_pred_class)
    recall = recall_score(y_test, y_pred_class)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Recall : {recall:.4f}')
    print('Confusion Matrix:')
    print(confusion_mat)