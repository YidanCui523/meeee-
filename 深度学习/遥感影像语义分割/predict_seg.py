import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

# ==================== 类别设置（与训练严格一致） ====================
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

class_colors = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (1.0, 0.5, 0.0),
    3: (0.6, 0.6, 0.6),
    4: (0.0, 0.0, 1.0),
    5: (1.0, 1.0, 0.0),
    6: (0.0, 1.0, 0.0),
    7: (0.5, 0.0, 0.5),
    8: (0.0, 1.0, 1.0),
    9: (1.0, 0.0, 1.0)
}


def mask_to_rgb(mask, class_colors):
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for cls, color in class_colors.items():
        rgb[mask == cls] = color
    return (rgb * 255).astype(np.uint8)


class UnlabeledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
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


def get_model(num_classes):
    local_weight_path = '20250530\model_supervised_best.pth'
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


def predict_and_export(model, dataloader, device, output_dir, csv_path=None, input_folder_name=None):
    import pandas as pd
    from skimage.measure import label, regionprops, perimeter

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    results = []
    # 移除 all_predictions 列表以节省内存
    class_counts = {i: 0 for i in range(len(class_map))}
    
    with torch.no_grad():
        for batch_idx, (imgs, names) in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}/{len(dataloader)}")
            imgs = imgs.to(device)
            outputs = model(imgs)['out']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for i, name in enumerate(names):
                pred = preds[i]
                
                # 直接统计类别分布，而不是存储所有预测
                unique_preds, pred_counts = np.unique(pred, return_counts=True)
                for cls, count in zip(unique_preds, pred_counts):
                    class_counts[cls] += count

                # 保存彩色分割
                rgb = mask_to_rgb(pred, class_colors)
                Image.fromarray(rgb).save(os.path.join(output_dir, f"{name}_seg.png"))

                # 统计
                areas = []
                perimeters = []
                for cls in range(len(class_map)):
                    binary_mask = (pred == cls).astype(np.uint8)
                    area = binary_mask.sum() / pred.size
                    areas.append(area)
                    labeled = label(binary_mask)
                    perim = sum(perimeter(region.image) for region in regionprops(labeled))
                    perimeters.append(perim)
                results.append([name] + areas + perimeters)
            
            # 清理GPU内存
            del imgs, outputs, preds
            torch.cuda.empty_cache()

    # 默认总是保存 CSV
    if csv_path is None:
        if input_folder_name:
            csv_path = os.path.join(output_dir, f'results_summary_{input_folder_name}.csv')
        else:
            csv_path = os.path.join(output_dir, 'results_summary.csv')
    import pandas as pd
    df = pd.DataFrame(results, columns=['image'] +
                      [f'class_{k}_ratio' for k in range(len(class_map))] +
                      [f'class_{k}_perimeter' for k in range(len(class_map))])
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV Saved to {csv_path}")

    # 输出总体预测分布
    print(f"Overall prediction distribution: {class_counts}")


def main():
    parser = argparse.ArgumentParser(description='Predict segmentation masks using a saved weight')
    parser.add_argument('--weights', default='20250530/model_supervised_best.pth')
    parser.add_argument('--img_dir', default='20250530/split_predict/part_3')
    parser.add_argument('--out_dir', default='20250530/outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 模型与权重
    model = get_model(num_classes=len(class_map)).to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    print(f"✅ Loaded weights: {args.weights}")

    # 数据
    ds = UnlabeledDataset(args.img_dir, transform=base_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 获取输入文件夹名称并用于CSV命名
    input_folder_name = os.path.basename(os.path.normpath(args.img_dir))
    csv_path = os.path.join(args.out_dir, f'results_summary_{input_folder_name}.csv')
    predict_and_export(model, loader, device, args.out_dir, csv_path, input_folder_name)
    print('✅ Prediction finished')


if __name__ == '__main__':
    main()



