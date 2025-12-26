
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.models.segmentation as models
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops, perimeter
import matplotlib.pyplot as plt

# =====================
# 1. 数据加载与掩码生成
# =====================
def labelme_to_mask(labelme_path, img_size, class_map):
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    with open(labelme_path, 'r') as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        cls_name = shape['label']
        if cls_name not in class_map:
            continue
        cls_id = class_map[cls_name]
        points = np.array(shape['points'], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], cls_id)
    return mask

class LabeledDataset(Dataset):
    def __init__(self, img_dir, labelme_dir, class_map, target_size=(467, 467)):
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        self.img_dir, self.labelme_dir = img_dir, labelme_dir
        self.class_map, self.target_size = class_map, target_size
    def __len__(self): return len(self.img_list)
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        labelme_path = os.path.join(self.labelme_dir, img_name.replace('.tif', '.json'))
        mask = labelme_to_mask(labelme_path, img.size, self.class_map)
        img = TF.to_tensor(TF.resize(img, self.target_size))
        mask = torch.from_numpy(np.array(TF.resize(Image.fromarray(mask), self.target_size, interpolation=Image.NEAREST))).long()
        return img, mask, img_name

class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, target_size=(467, 467)):
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        self.img_dir, self.target_size = img_dir, target_size
    def __len__(self): return len(self.img_list)
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        img = TF.to_tensor(TF.resize(img, self.target_size))
        return img, img_name

# =====================
# 2. 模型与训练
# =====================
def get_model(num_classes):
    return models.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        imgs, masks = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs)['out'], masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def generate_pseudo_labels(model, dataloader, device):
    pseudo_labels = {}
    model.eval()
    with torch.no_grad():
        for imgs, img_names in dataloader:
            preds = torch.argmax(model(imgs.to(device))['out'], dim=1).cpu()
            for i, name in enumerate(img_names): pseudo_labels[name] = preds[i]
    return pseudo_labels

# =====================
# 3. 特征提取与聚类
# =====================
def extract_features(model, dataloader, device):
    model.eval()
    features = {}
    with torch.no_grad():
        for imgs, img_names in dataloader:
            feats = model.backbone(imgs.to(device))['out'].mean(dim=(2,3)).cpu().numpy()
            for i, name in enumerate(img_names): features[name] = feats[i]
    return features

def cluster_features(features_dict, num_clusters):
    names = list(features_dict.keys())
    mat = np.array([features_dict[n] for n in names])
    labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(mat)
    return dict(zip(names, labels))

def filter_pseudo_labels_by_consistency(pseudo_labels, cluster_labels):
    selected = {}
    for name, mask in pseudo_labels.items():
        dominant_class = np.bincount(mask.flatten()).argmax()
        if dominant_class == cluster_labels[name]:
            selected[name] = mask
    return selected

class ConsistentPseudoDataset(Dataset):
    def __init__(self, unlabeled_dataset, selected_pseudo_labels):
        self.dataset = unlabeled_dataset
        self.labels = selected_pseudo_labels
        self.valid_names = list(self.labels.keys())
    def __len__(self): return len(self.valid_names)
    def __getitem__(self, idx):
        name = self.valid_names[idx]
        img, _ = self.dataset[self.dataset.img_list.index(name)]
        return img, self.labels[name], name

def prepare_mixed_training_data(model, device, unlabeled_loader, pseudo_labels, unlabeled_dataset, labeled_dataset, class_map):
    features = extract_features(model, unlabeled_loader, device)
    cluster_labels = cluster_features(features, num_clusters=len(class_map))
    filtered = filter_pseudo_labels_by_consistency(pseudo_labels, cluster_labels)
    consistent_dataset = ConsistentPseudoDataset(unlabeled_dataset, filtered)
    mixed_dataset = torch.utils.data.ConcatDataset([labeled_dataset, consistent_dataset])
    return DataLoader(mixed_dataset, batch_size=4, shuffle=True)

# =====================
# 4. 主程序入口
# =====================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_map = {'building': 0, 'railway': 1, 'road': 2, 'water': 3, 'instrument': 4,
                 'green': 5, 'square': 6, 'undevelopment': 7, 'playground': 8}
    labeled_dataset = LabeledDataset('data/images_labeled', 'data/geojson_labeled', class_map)
    unlabeled_dataset = UnlabeledDataset('split_predict/part_3')
    labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False)
    
    model = get_model(len(class_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("🔧 Stage 1: Supervised training")
    for epoch in range(5):
        loss = train_one_epoch(model, labeled_loader, optimizer, criterion, device)
        print(f"[Supervised] Epoch {epoch+1}: Loss={loss:.4f}")

    print("🔧 Stage 2: Generating pseudo-labels")
    pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, device)

    print("🔧 Stage 3: Filtering pseudo-labels with clustering")
    mixed_loader = prepare_mixed_training_data(model, device, unlabeled_loader, pseudo_labels,
                                               unlabeled_dataset, labeled_dataset, class_map)
    for epoch in range(5):
        loss = train_one_epoch(model, mixed_loader, optimizer, criterion, device)
        print(f"[Semi-Supervised] Epoch {epoch+1}: Loss={loss:.4f}")
