import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import time
from label_to_list import label_list
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

versions = ['sr', 'br', 'lr', 'gr']

def extract_feature(img_path, model, device): # device variable only for read, so global, no need to get from param
        img = Image.open(img_path).convert('RGB') 
        img = transform(img).unsqueeze(0).to(device) # insert batch dim (n,m) -> (1,n,m) for model inference
        with torch.no_grad(): # use model without gradient update, only for inference
            feature_dict= model.forward_features(img) # shape : (1, feature_dim)
            feature = feature_dict['x_norm_clstoken']
        return feature.squeeze(0).cpu().numpy() # shape : (feature_dim,). Because of torch.no_grad(), no need to .detach()

# Regression
class LogisticRegression(nn.Module):
    def __init__(self,feature_dim):
        super().__init__()
        self.weight = nn.Linear(feature_dim,2)
    
    def forward(self, x):
        return self.weight(x) # (batch,3) with non fire class
    
if __name__=="__main__":
    for v in versions:
        VERSION = v
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device : {device}")

        #if not os.path.exists('x_feature.npy'):
        # if VERSION=='s':
        #     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        # elif VERSION=='b':
        #     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        # elif VERSION=='l':
        #     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
        # elif VERSION=='g':
        #     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
        if VERSION=='sr':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
        elif VERSION=='br':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device)
        elif VERSION=='lr':
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
        else:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
        model.eval()
            
        train_fire_imgs = r"F:\dataset\YOLO11_fire_dataset\train\sorted\fire"
        train_smoke_imgs = r"F:\dataset\YOLO11_fire_dataset\train\sorted\smoke"
        train_none_imgs = r"F:\dataset\YOLO11_fire_dataset\train\sorted\none" 

        train_x = []
        train_label = []

        tic = time.time()

        #fire img
        for i,train_img in enumerate(os.listdir(train_fire_imgs)):
            path = os.path.join(train_fire_imgs, train_img)
            train_x.append(extract_feature(path, model, device))
            train_label.append(0)
            print(f"(Training) Fire image feature extract {i}")
        
        #smoke img
        for i,train_img in enumerate(os.listdir(train_smoke_imgs)):
            path = os.path.join(train_smoke_imgs, train_img)
            train_x.append(extract_feature(path, model, device))
            train_label.append(1)
            print(f"(Training) Smoke image feature extract {i}")

        # #none img
        # for i,train_img in enumerate(os.listdir(train_none_imgs)):
        #     path = os.path.join(train_none_imgs, train_img)
        #     train_x.append(extract_feature(path, model, device))
        #     train_label.append(0)
        #     print(f"(Training) None image feature extract {i}")

        tocFeat = time.time()

        train_x = np.array(train_x)
        train_label = np.array(train_label)

        # save features
        np.save('train_x_feature.npy', train_x)
        np.save('train_label.npy',train_label)

        # else:
        #     x = np.load('x_feature.npy')
        #     label = np.load('label.npy')

        print(f"feature shape : {train_x.shape}, Labels : {train_label.shape}")
        print(f"x max : {np.max(train_x)}, x min : {np.min(train_x)}")

        # to device
        X_train = torch.tensor(np.array(train_x), dtype=torch.float32).to(device)
        y_train = torch.tensor(np.array(train_label), dtype=torch.long).to(device)
        
        classifier = LogisticRegression(train_x.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.01)

        epochs = 1000

        for epoch in range(epochs):
            classifier.train()
            optimizer.zero_grad()
            train_pred = classifier(X_train)
            loss = criterion(train_pred, y_train)
            loss.backward()
            optimizer.step()
            print(f"epoch : {epoch} | loss : {loss.item():.4f}")

        # save parameter
        torch.save(classifier.state_dict(), f'version_{v}_classifier_weights.pth')
        tocEnd = time.time()

        # Evaluate
        test_fire_imgs = r'F:\dataset\YOLO11_fire_dataset\test\sorted\fire'
        test_smoke_imgs = r'F:\dataset\YOLO11_fire_dataset\test\sorted\smoke'
        test_none_imgs = r'F:\dataset\YOLO11_fire_dataset\test\sorted\none'

        test_label = []
        test_x = []

        #fire img
        for i,test_img in enumerate(os.listdir(test_fire_imgs)):
            path = os.path.join(test_fire_imgs, test_img)
            test_x.append(extract_feature(path, model, device))
            test_label.append(0)
            print(f"(Testing) Fire image feature extract {i}")
        
        #smoke img
        for i,test_img in enumerate(os.listdir(test_smoke_imgs)):
            path = os.path.join(test_smoke_imgs, test_img)
            test_x.append(extract_feature(path, model, device))
            test_label.append(1)
            print(f"(Testing) Smoke image feature extract {i}")

        # #none img
        # for i,test_img in enumerate(os.listdir(test_none_imgs)):
        #     path = os.path.join(test_none_imgs, test_img)
        #     test_x.append(extract_feature(path, model, device))
        #     test_label.append(0)
        #     print(f"(Testing) None image feature extract {i}")

        X_test = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
        y_test = torch.tensor(np.array(test_label), dtype=torch.long) # CrossEntropyLoss 는 long type을 받음

        classifier.eval()
        with torch.no_grad():
            y_pred = classifier(X_test).detach().cpu().numpy()
            y_pred_class = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_class)
        f1 = f1_score(y_test,y_pred_class)
        confusion_mat = confusion_matrix(y_test,y_pred_class)
        recall = recall_score(y_test, y_pred_class)
        training_time = tocEnd - tic
        feature_time = tocFeat - tic
        print(f"Total training taked : {training_time//60} : {training_time%60}")
        print(f"Feature extracting taked : {feature_time//60} : {feature_time%60}")
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Recall : {recall:.4f}')
        print('Confusion Matrix:')
        print(confusion_mat)

        mode = "w" if v=='s' else "a"
        with open("Train_Log.txt", mode, encoding="utf-8") as f:
            f.write(f"Model : {v}\n")
            f.write(f"Total training taked : {int(training_time//60)} min {training_time%60:.5f} sec\n")
            f.write(f"Feature extracting taked : {int(feature_time//60)} min {feature_time%60:.5f} sec\n")
            f.write(f'Accuracy : {accuracy:.4f}\n')
            f.write(f'F1-score : {f1:.4f}\n')
            f.write(f'Recall : {recall:.4f}\n')
            f.write('Confusion Matrix:\n')
            f.write(str(confusion_mat))
            f.write('\n\n')
        
            


