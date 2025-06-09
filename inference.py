import torch
from train import extract_feature, LogisticRegression
import torchvision.transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

from vision_transformer import VisionTransformer

INFERENCE_EACH = False

if not INFERENCE_EACH:
    img_path = r"sample_data"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device : {device}")

    time_tic_load = time.time()
    DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
    time_toc_load = time.time()
    DINO.eval()

    # Bounding Box
    #Vit = VisionTransformer().to(device)


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

    # Bounding Box
    #last_attention = []

    time_tic_DINO = time.time()
    for img in os.listdir(img_path):
        path = os.path.join(img_path, img)
        img = Image.open(path).convert('RGB') 
        img_arr = np.array(img)
        x.append(extract_feature(path, DINO, device))

        # Bounding Box
        # img_bb = transform(img).unsqueeze(0).to(device)
        # last_attention.append(Vit.get_last_selfattention(img_bb))

        images.append(img_arr)

    #print(f"last_attention : {last_attention[0].shape}") # (1,12,197,197) (batch, head, token + cls, token + cls)
    
    # try for just one image
    def bbox(last_attention, image, threshold=0.6):
        h, w, _ = image.shape

        att_map = last_attention[0,:,0,1:]
        att_mean = att_map.mean(0).reshape(14,14)
        att_mean = (att_mean - att_mean.min())/(att_mean.max() - att_mean.min())

        mask = (att_mean > threshold).to(torch.uint8).cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cnt) for cnt in contours]

        bbox_img = image.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(bbox_img, (x,y), (x+bw, y+bh), (0, 255, 0), 2)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(bbox_img)
        plt.axis('off')
        plt.title("Attention-based Bounding Box")
        plt.show()

    #bbox(last_attention[0], images[0], threshold=0.8)

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




    # plot result
    # fig, axes = plt.subplots(2,5, figsize=(20,10))
    # idx = 0
    # for i in range(2):
    #     for j in range(5):
    #         sample_img = cv2.resize(images[idx], (200,200))
    #         sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
    #         axes[i][j].imshow(sample_img)
    #         axes[i][j].axis('off')
    #         axes[i][j].set_title(f"label : {label[idx]}\n pred : {pred_class[idx]}", fontsize=10)
    #         idx += 1
    # plt.tight_layout()
    # plt.show()

else:
    img_path = r"sample_data"
    img_list = [os.path.join(img_path, img) for img in os.listdir(img_path)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device : {device}")

    avg_time = {"hour" : [], "min" : [], "sec" : [], "tic-toc" : []}
    avg_extfeat = {"hour" : [], "min" : [], "sec" : [], "tic-toc" : []}
    avg_log = {"hour" : [], "min" : [], "sec" : [], "tic-toc" : []}
    
    label = [1,1,1,1,1,0,0,0,0,0]

    time_tic_load = time.time()
    DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
    time_toc_load = time.time()
    DINO.eval()
    
    dhour = int((time_toc_load-time_tic_load)//3600)
    dmint = int(((time_toc_load-time_tic_load)%3600)//60)
    dsec = (time_toc_load-time_tic_load)%60 
    dtic = (time_toc_load-time_tic_load)*1000
    print(f"Loading DINO : {dhour}:{dmint}:{format(dsec, '4.2f')} ({(time_toc_load-time_tic_load)*1000:.2f} ms) Done")

    for idx,img in enumerate(img_list):
        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        x=[]
        images = []
        path = img
        time_tic_DINO = time.time()
        img = Image.open(path).convert('RGB') 
        img = np.array(img)
        x.append(extract_feature(path, DINO, device))
        images.append(img)
        time_toc_DINO = time.time()

        avg_extfeat['hour'].append(int((time_toc_DINO-time_tic_DINO)//3600))
        avg_extfeat['min'].append(int(((time_toc_DINO-time_tic_DINO)%3600)//60))
        avg_extfeat['sec'].append((time_toc_DINO-time_tic_DINO)%60)
        avg_extfeat['tic-toc'].append((time_toc_DINO-time_tic_DINO)*1000)
        
        x = torch.tensor(np.array(x), dtype=torch.float32)
        print(x.shape)

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

        print(f"label : {label[idx]} / prediction : {pred_class[0]}")

        avg_log['hour'].append(int((time_toc_log-time_tic_log)//3600))
        avg_log['min'].append(int(((time_toc_log-time_tic_log)%3600)//60))
        avg_log['sec'].append((time_toc_log-time_tic_log)%60)
        avg_log['tic-toc'].append((time_toc_log-time_tic_log)*1000) 

        avg_time['hour'].append(int((time_toc_log-time_tic_DINO)//3600))
        avg_time['min'].append(int(((time_toc_log-time_tic_DINO)%3600)//60))
        avg_time['sec'].append((time_toc_log-time_tic_DINO)%60 )
        avg_time['tic-toc'].append((time_toc_log-time_tic_DINO)*1000)

    avg_hour = np.mean(avg_time['hour']) + dhour
    avg_min = np.mean(avg_time['min']) + dmint
    avg_sec = np.mean(avg_time['sec']) + dsec
    avg_tic = np.mean(avg_time['tic-toc']) + dtic

    avg_extfeat_hour = np.mean(avg_extfeat['hour'])
    avg_extfeat_min = np.mean(avg_extfeat['min'])
    avg_extfeat_sec = np.mean(avg_extfeat['sec'])
    avg_extfeat_tic = np.mean(avg_extfeat['tic-toc'])

    avg_log_hour = np.mean(avg_log['hour'])
    avg_log_min = np.mean(avg_log['min'])
    avg_log_sec = np.mean(avg_log['sec'])
    avg_log_tic = np.mean(avg_log['tic-toc'])

    print("------------------------------------")
    print(f"(average) Total Inference : {avg_hour}:{avg_min}:{format(avg_sec, '4.2f')} ({avg_tic:.2f} ms) Done")
    print("------------------------------------")
    print(f"Loading DINO : {dhour}:{dmint}:{format(dsec, '4.2f')} ({dtic:.2f} ms) Done")
    print(f"(average) DINO extracting features : {avg_extfeat_hour}:{avg_extfeat_min}:{format(avg_extfeat_sec, '4.2f')} ({avg_extfeat_tic:.2f} ms) Done")
    print(f"(average) Logistic Regrssion Classification : {avg_log_hour}:{avg_log_min}:{format(avg_log_sec, '4.2f')} ({avg_log_tic:.2f} ms) Done")