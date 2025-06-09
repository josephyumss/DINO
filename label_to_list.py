import os

def label_list(path):
    label = []
    bbox = []
    for txt in os.listdir(path):
        with open(txt,'r') as f:
            values = []
            for line in f:
                text = list(map(float, line.strip().split()))
                values.extend(text)
        
    label.append(values[0])
    bbox.append([values[1:]])

    return label, bbox