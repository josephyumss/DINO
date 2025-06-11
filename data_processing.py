import os
import shutil

imgs_path = r'F:\dataset\YOLO11_fire_dataset\test\images'
label_path = r'F:\dataset\YOLO11_fire_dataset\test\labels'
output_path = r'F:\dataset\YOLO11_fire_dataset\test\sorted'

class_names = {
    0: 'fire',
    1: 'smoke',
    2: 'none'
}

for cname in class_names.values():
    os.makedirs(os.path.join(output_path, cname), exist_ok=True)

for label_file in os.listdir(label_path):
    label_file_full_path = os.path.join(label_path, label_file)
    with open(label_file_full_path, 'r') as f:
        for line in f:
            text = list(map(float, line.strip().split()))
        img_class = text[0]
        class_name = class_names.get(img_class, 'unknown')
    
    file_name = os.path.splitext(label_file)[0]
    img_path = os.path.join(imgs_path,file_name+'.jpg')
    dest_path = os.path.join(output_path, class_name, file_name+'.jpg')
    shutil.copy(img_path, dest_path)