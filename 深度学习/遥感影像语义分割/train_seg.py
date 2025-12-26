import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dcrf_utils
import argparse


# ImageNet 归一化常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ==================== 数据增强 ====================
base_transform = A.Compose([
    A.Resize(467, 467),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

augment_transform = A.Compose([
    A.Resize(467, 467),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.00),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# ==================== 类别设置（含背景） ====================
class_map = {
    'background': 0,
    'building': 1,
    'railway': 2,
    'road': 3,
    'water': 4,
    'instrument': 5,
    'green': 6,
    'square': 7,
    'undevelopment': 8,
    'playground': 9
}


def labelme_to_mask(labelme_path, img_size, class_map):
    mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    with open(labelme_path, 'r') as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        cls_name = shape['label']
        if cls_name in class_map:
            points = np.array(shape['points'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], class_map[cls_name])
    return mask


# ==================== 数据集 ====================
class LabeledDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir)
                         if f.endswith('.tif') and os.path.exists(os.path.join(label_dir, f.replace('.tif', '.json')))]
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.tif', '.json'))

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = labelme_to_mask(label_path, image.shape[:2][::-1], class_map)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'].float(), transformed['mask'].long(), img_name


class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, img_name)).convert('RGB'))
        if self.transform:
            transformed = self.transform(image=image)
            return transformed['image'].float(), img_name
        else:
            return (torch.from_numpy(image).permute(2, 0, 1).float() / 255.0), img_name


class PseudoLabeledDataset(Dataset):
    def __init__(self, unlabeled_dataset, pseudo_labels):
        self.unlabeled_dataset = unlabeled_dataset
        self.pseudo_labels = pseudo_labels
    def __len__(self):
        return len(self.unlabeled_dataset)
    def __getitem__(self, idx):
        img, name = self.unlabeled_dataset[idx]
        return img, self.pseudo_labels[name], name


# ==================== 损失与模型 ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


def get_model(num_classes):
    local_weight_path = '20250530/deeplabv3_resnet101_coco-586e9e4e.pth'
    model = deeplabv3_resnet101(pretrained=False)

    if os.path.exists(local_weight_path):
        try:
            checkpoint = torch.load(local_weight_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ 已加载本地预训练权重: {local_weight_path}")
        except Exception as e:
            print(f"⚠️ 加载本地权重失败，将使用随机初始化骨干。原因: {str(e)}")
    else:
        print(f"⚠️ 未找到本地权重文件: {local_weight_path}，将使用随机初始化骨干。")

    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ==================== 伪标签 + CRF ====================
def apply_crf(image, prediction):
    prediction = np.clip(prediction, 0, len(class_map) - 1)
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    unary = dcrf_utils.unary_from_labels(prediction.astype(np.uint8), len(class_map), gt_prob=0.7, zero_unsure=False)
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], len(class_map))
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape(image.shape[:2])


def generate_pseudo_labels(model, dataloader, device):
    pseudo_labels = {}
    model.eval()
    with torch.no_grad():
        for imgs, names in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)['out']
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            for i, name in enumerate(names):
                img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * np.array(IMAGENET_STD)[None, None, :] + np.array(IMAGENET_MEAN)[None, None, :])
                img_np = np.clip(img_np, 0.0, 1.0)
                img_np = (img_np * 255).astype(np.uint8)
                mask_np = preds[i].cpu().numpy()
                if max_probs[i].mean() <= 0.2:
                    mask_np[:] = 0
                mask_crf = apply_crf(img_np, mask_np)
                pseudo_labels[name] = torch.tensor(mask_crf).long()
    return pseudo_labels


# ==================== 训练 ====================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks, *_ in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model (with optional semi-supervised)')
    parser.add_argument('--train_img_dir', default='20250530/data/images_labeled')
    parser.add_argument('--train_label_dir', default='20250530/data/geojson_labeled')
    parser.add_argument('--unlabeled_dir', default='20250530/split_predict/part_1')
    parser.add_argument('--save_dir', default='20250530')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sup_epochs', type=int, default=20)
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--semi_epochs', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    labeled_base = LabeledDataset(args.train_img_dir, args.train_label_dir, base_transform)
    labeled_aug = LabeledDataset(args.train_img_dir, args.train_label_dir, augment_transform)
    full_labeled = ConcatDataset([labeled_base, labeled_aug])

    unlabeled = UnlabeledDataset(args.unlabeled_dir, transform=base_transform)
    labeled_loader = DataLoader(full_labeled, batch_size=args.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled, batch_size=args.batch_size, shuffle=False)

    model = get_model(num_classes=len(class_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss()

    print('Stage 1: Supervised training')
    best_loss = float('inf')
    for epoch in range(args.sup_epochs):
        loss = train_one_epoch(model, labeled_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{args.sup_epochs} - Loss: {loss:.4f}')
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_supervised_best.pth'))
            print('✅ Saved best supervised model: model_supervised_best.pth')
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_supervised_final.pth'))
    print('✅ Saved final supervised model: model_supervised_final.pth')

    if args.semi and len(unlabeled) > 0:
        print('Stage 2: Generate pseudo labels')
        pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, device)
        pseudo_dataset = PseudoLabeledDataset(unlabeled, pseudo_labels)
        combined_loader = DataLoader(ConcatDataset([full_labeled, pseudo_dataset]), batch_size=args.batch_size, shuffle=True)

        print('Stage 3: Semi-supervised training')
        best_semi_loss = float('inf')
        for epoch in range(args.semi_epochs):
            loss = train_one_epoch(model, combined_loader, optimizer, criterion, device)
            print(f'Semi Epoch {epoch+1}/{args.semi_epochs} - Loss: {loss:.4f}')
            if loss < best_semi_loss:
                best_semi_loss = loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_semisupervised_best.pth'))
                print('✅ Saved best semi-supervised model: model_semisupervised_best.pth')
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_semisupervised_final.pth'))
        print('✅ Saved final semi-supervised model: model_semisupervised_final.pth')

    print('Done training.')


if __name__ == '__main__':
    main()


