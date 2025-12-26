import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加当前目录到路径
sys.path.append('.')

# 导入必要的函数和类
from 0705 import UnlabeledDataset, get_model, mask_to_rgb, class_colors, class_map

def test_visualization():
    print("Testing visualization function...")
    
    # 检查数据集
    dataset_path = '20250530/split_predict/part_try20'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # 创建数据集
    dataset = UnlabeledDataset(dataset_path, transform=None)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    # 测试加载一个样本
    try:
        img, name = dataset[0]
        print(f"✅ Successfully loaded sample: {name}")
        print(f"Image shape: {img.shape}")
        print(f"Image type: {type(img)}")
        print(f"Image range: {img.min():.3f} to {img.max():.3f}")
    except Exception as e:
        print(f"❌ Error loading sample: {str(e)}")
        return
    
    # 创建模型（如果GPU可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model = get_model(num_classes=len(class_map)).to(device)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return
    
    # 测试模型推理
    try:
        model.eval()
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            outputs = model(img_tensor)['out']
            print(f"✅ Model inference successful")
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: {outputs.min():.4f} to {outputs.max():.4f}")
            
            pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            print(f"Prediction shape: {pred.shape}")
            print(f"Prediction unique values: {np.unique(pred)}")
            
            # 测试mask_to_rgb函数
            rgb = mask_to_rgb(pred, class_colors)
            print(f"RGB mask shape: {rgb.shape}")
            
    except Exception as e:
        print(f"❌ Error during model inference: {str(e)}")
        return
    
    # 测试保存图像
    try:
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.title('Prediction')
        plt.axis('off')
        
        test_path = '20250530/test_visualization.png'
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_path):
            print(f"✅ Test image saved successfully: {test_path}")
        else:
            print(f"❌ Test image was not saved")
            
    except Exception as e:
        print(f"❌ Error saving test image: {str(e)}")
        return
    
    print("✅ All tests passed! Visualization function should work.")

if __name__ == '__main__':
    test_visualization()
