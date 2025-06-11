import os

def label_list(path):
    label = []
    bbox = []
    for idx,txt in enumerate(os.listdir(path)):
        txt = os.path.join(path,txt)
        with open(txt,'r') as f:
            for line in f:
                text = list(map(float, line.strip().split()))
            label.append(text[0])
            bbox.append([text[1:]])
    return label, bbox