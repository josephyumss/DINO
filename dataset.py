import os
from torch.utils.data import Dataset
from PIL import Image

class imgDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.label_map = {
            'fire' : 0,
            'smoke' : 1
        }

        self.data = []
        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            for data in os.listdir(label_path):
                data_path = os.path.join(label_path, data)
                self.data.append((data_path, self.label_map[label]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label