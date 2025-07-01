import torch
import torch.nn as nn
import torch.optim as optim
from dataset import imgDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

class finetuned_dino(nn.Module):
    def __init__(self, backbone, feature_dim = 384, num_classes = 3): # dino s' dim 384
        super().__init__()
        self.dino = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        feature = self.dino.forward_features(x)['x_norm_clstoken']
        return self.classifier(feature)
    
if __name__=='__main__':  
    models = ['dinov2_vits14_reg','dinov2_vitb14_reg']

    for version in models:
        transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ])

        train_dir = r"C:\Users\User\dataset\roboflow_dataset\train\sorted"
        train_data = imgDataset(root=train_dir, transform=transform)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device : {device}")

        dino = torch.hub.load('facebookresearch/dinov2', version).to(device)
        model = finetuned_dino(dino).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([
            {"params" : model.dino.parameters(), "lr" : 1e-5},
            {"params" : model.classifier.parameters(), "lr" : 1e-3}])
        epochs = 10

        model.train()
        train_tic = time.time()
        for epoch in range(epochs):
            tloss = 0
            for img, lab in train_loader:
                img, lab = img.to(device), lab.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, lab)
                optimizer.step()
                tloss += loss.item()

            print(f"Epoch {epoch} - Loss : {tloss:.4f}")
        train_toc = time.time()

        #train time
        train_time = train_toc-train_tic
        
        torch.save(model.state_dict(), f'fine_{version}_weights.pth')

        test_dir = r"C:\Users\User\dataset\roboflow_dataset\test\sorted"
        test_data = imgDataset(root=test_dir, transform=transform)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

        model.eval()
        y_true = []
        y_pred = []
        inf_time = []

        with torch.no_grad():
            for img, lab in test_loader:
                img, lab = img.to(device), lab.to(device)

                inf_tic = time.time()
                output = model(img)
                inf_toc = time.time()

                pred = output.argmax(dim=1)

                y_true.extend(lab.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                inf_time.append(inf_toc-inf_tic)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)

        # inference time
        avg_inf_time = sum(inf_time)/len(inf_time)

        # w mode는 파일 내용 초기화 후, 작성
        # a mode는 이어서 작성
        with open("dino_fine_Log.txt", 'a', encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write(f"Model : {version}\n")
                f.write(f"Total training taked : {int(train_time/60)} min {train_time%60:.5f} sec ({train_time:.2f} ms))\n")
                f.write(f"(Average) Inference taked : {int(inf_time//60)} min {inf_time%60:.5f} sec ({inf_time:.2f} ms)\n")
                f.write(f'Accuracy : {accuracy:.4f}\n')
                f.write(f'F1-score : {f1:.4f}\n')
                f.write(f'Recall : {recall:.4f}\n')
                f.write('Confusion Matrix:\n')
                f.write(str(cfm))
                f.write('\n\n')