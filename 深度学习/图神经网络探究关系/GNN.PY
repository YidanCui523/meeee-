import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
try:
    import shap
    _shap_available = True
except Exception:
    _shap_available = False

# =========================
# 1. 读取数据
# =========================
csv_file = "regession\play2019_0906.csv"  # 你的CSV路径
df = pd.read_csv(csv_file)

# 特征列（形指标）
shape_features = [
    "building_ratio0", "railway_ratio1", "road_ratio2", "water_ratio3",
    "instru_ratio4", "green_ratio5", "square_ratio6", "undevelop_ratio7",
    "playground_ratio8", "LPI_Max", "PR_COUNTclass", "SHI_zhouchang",
    "RND_road", "Social", "Life", "Entertainment",
    "Production", "POI_density", "MAX_Elevation", "MEAN_Elevation",
]

# 目标列（流入、流出、betweenness）
target_cols = ["PLAY_D_IN", "PLAY_O_OUT", "PLAY__Betweenness"]

coords = df[["lon", "lat"]].values
X = df[shape_features].values
Y = df[target_cols].values

# =========================
# 2. 数据预处理 - 自变量已经标准化，直接使用
# =========================
# 自变量已经标准化，直接使用
X_processed = X.copy()
# 不对目标变量进行标准化，保持原始值
Y_original = Y.copy()

# =========================
# 3. 建图（空间邻居 + 边权）
# =========================
tree = BallTree(np.deg2rad(coords), metric='haversine')
radius = 2.0 / 6371.0  # 半径 0.5km (地球半径单位)

edges = []
edge_weights = []
for i, coord in enumerate(np.deg2rad(coords)):
    idx = tree.query_radius([coord], r=radius)[0]
    for j in idx:
        if i != j:
            dist = np.linalg.norm(coords[i] - coords[j])
            edges.append((i, j))
            edge_weights.append(1 / (dist + 1e-6))  # 距离倒数做权重

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float)

# =========================
# 4. 构建 PyG 数据对象
# =========================
data = Data(
    x=torch.tensor(X_processed, dtype=torch.float),
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=torch.tensor(Y_original, dtype=torch.float)
)

# =========================
# 5. 定义改进的 GNN 模型
# =========================
class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImprovedGCN, self).__init__()
        
        # 第一层GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        
        # 第二层GCN
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        # 第三层GCN
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn3 = BatchNorm(hidden_channels // 2)
        
        # 第四层GCN
        self.conv4 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.bn4 = BatchNorm(hidden_channels // 4)
        
        # 全连接层
        self.fc1 = torch.nn.Linear(hidden_channels // 4, hidden_channels // 2)
        self.fc2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc3 = torch.nn.Linear(hidden_channels // 4, out_channels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # 输出激活函数确保非负
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # 第一层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第四层
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # 使用ReLU确保输出非负
        return self.relu(x)

# 定义学习率调度器
class CosineAnnealingScheduler:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 定义自定义损失函数（结合MSE和Huber损失）
def combined_loss(pred, target, alpha=0.7):
    """
    结合MSE和Huber损失，对异常值更鲁棒
    """
    mse_loss = F.mse_loss(pred, target)
    huber_loss = F.smooth_l1_loss(pred, target)
    return alpha * mse_loss + (1 - alpha) * huber_loss

# 创建模型
model = ImprovedGCN(in_channels=X.shape[1], hidden_channels=64, out_channels=3, dropout=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
scheduler = CosineAnnealingScheduler(optimizer, T_max=3000, eta_min=1e-6)
loss_fn = combined_loss

# =========================
# 6. 训练
# =========================
print("开始训练改进的GNN模型...")
best_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = loss_fn(pred, data.y)
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step(epoch)
    
    # 早停机制
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_gnn_model.pth')
    else:
        patience_counter += 1
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 早停
    if patience_counter >= patience:
        print(f"早停在第 {epoch} 轮")
        break

# 加载最佳模型
model.load_state_dict(torch.load('best_gnn_model.pth'))

# =========================
# 7. 预测
# =========================
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).numpy()

df["pred_in"], df["pred_out"], df["pred_between"] = pred[:, 0], pred[:, 1], pred[:, 2]

# 检查预测值的范围
print(f"\n预测结果:")
print(f"预测流入最小值: {pred[:, 0].min():.2f}, 最大值: {pred[:, 0].max():.2f}")
print(f"预测流出最小值: {pred[:, 1].min():.2f}, 最大值: {pred[:, 1].max():.2f}")
print(f"预测betweenness最小值: {pred[:, 2].min():.2f}, 最大值: {pred[:, 2].max():.2f}")

df.to_csv("regession/output/play2019_0906.csv", index=False)
print("✅ 改进模型的预测结果已保存到 play2019_0906.csv")

# =========================
# 创建格网基础结果表
# =========================
print("\n创建格网基础结果表...")

# 创建格网基础结果表
grid_results_data = []

for i in range(len(df)):
    grid_data = {
        'id': df.iloc[i]['id2019'],  # 使用原始数据中的id2019作为格网编号
        'lon': df.iloc[i]['lon'],  # 经度坐标
        'lat': df.iloc[i]['lat']   # 纬度坐标
    }
    
    # 添加所有形态特征值
    for feature_name in shape_features:
        grid_data[feature_name] = df.iloc[i][feature_name]
    
    # 添加目标变量（可选）
    for target_name in target_cols:
        grid_data[target_name] = df.iloc[i][target_name]
    
    # 添加预测值（可选）
    grid_data['pred_in'] = df.iloc[i]['pred_in']
    grid_data['pred_out'] = df.iloc[i]['pred_out']
    grid_data['pred_between'] = df.iloc[i]['pred_between']
    
    grid_results_data.append(grid_data)

# 转换为DataFrame并保存
grid_results_df = pd.DataFrame(grid_results_data)
grid_results_file = "regession/output/grid_results.csv"

# 尝试保存文件，如果权限被拒绝则使用备用文件名
try:
    grid_results_df.to_csv(grid_results_file, index=False)
    print(f"✅ 格网基础结果表已保存到: {grid_results_file}")
except PermissionError:
    # 如果权限被拒绝，使用带时间戳的备用文件名
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"regession/output/grid_results_{timestamp}.csv"
    grid_results_df.to_csv(backup_file, index=False)
    print(f"⚠️  原文件被占用，已保存到备用文件: {backup_file}")
    grid_results_file = backup_file
except Exception as e:
    print(f"❌ 保存文件时发生错误: {e}")
    # 尝试保存到当前目录
    fallback_file = "grid_results_fallback.csv"
    grid_results_df.to_csv(fallback_file, index=False)
    print(f"📁 已保存到当前目录: {fallback_file}")
    grid_results_file = fallback_file

print(f"   包含 {len(grid_results_df)} 个格网")
print(f"   包含 {len(grid_results_df.columns)} 个字段")
print(f"   字段包括: id, lon, lat, {len(shape_features)} 个形态特征, {len(target_cols)} 个目标变量, 3个预测值")

# =========================
# 8. 可视化
# =========================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(df["lon"], df["lat"], c=df["pred_in"], cmap="viridis", s=10)
plt.colorbar(label="Predicted Inflow")
plt.title("Improved GNN - Predicted Inflow Heatmap")

plt.subplot(1, 3, 2)
plt.scatter(df["lon"], df["lat"], c=df["pred_out"], cmap="viridis", s=10)
plt.colorbar(label="Predicted Outflow")
plt.title("Improved GNN - Predicted Outflow Heatmap")

plt.subplot(1, 3, 3)
plt.scatter(df["lon"], df["lat"], c=df["pred_between"], cmap="viridis", s=10)
plt.colorbar(label="Predicted Betweenness")
plt.title("Improved GNN - Predicted Betweenness Heatmap")

plt.tight_layout()
plt.savefig('regession/output/basic_prediction_heatmaps_play.png', dpi=300, bbox_inches='tight')

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 读取预测结果
df = pd.read_csv('regession/output/play2019_0906.csv')

# 计算R²分数
r2_in = r2_score(df['PLAY_D_IN'], df['pred_in'])
r2_out = r2_score(df['PLAY_O_OUT'], df['pred_out'])
r2_between = r2_score(df['PLAY__Betweenness'], df['pred_between'])

# 计算MAE
mae_in = mean_absolute_error(df['PLAY_D_IN'], df['pred_in'])
mae_out = mean_absolute_error(df['PLAY_O_OUT'], df['pred_out'])
mae_between = mean_absolute_error(df['PLAY__Betweenness'], df['pred_between'])

# 计算RMSE
rmse_in = np.sqrt(mean_squared_error(df['PLAY_D_IN'], df['pred_in']))
rmse_out = np.sqrt(mean_squared_error(df['PLAY_O_OUT'], df['pred_out']))
rmse_between = np.sqrt(mean_squared_error(df['PLAY__Betweenness'], df['pred_between']))

print("=== 模型性能评估 ===")
print(f"R² 分数:")
print(f"  流入预测: {r2_in:.4f}")
print(f"  流出预测: {r2_out:.4f}")
print(f"  Betweenness预测: {r2_between:.4f}")

print(f"\nMAE (平均绝对误差):")
print(f"  流入预测: {mae_in:.2f}")
print(f"  流出预测: {mae_out:.2f}")
print(f"  Betweenness预测: {mae_between:.2f}")

print(f"\nRMSE (均方根误差):")
print(f"  流入预测: {rmse_in:.2f}")
print(f"  流出预测: {rmse_out:.2f}")
print(f"  Betweenness预测: {rmse_between:.2f}")

print(f"\n预测值统计:")
print(f"  流入: 最小值={df['pred_in'].min():.2f}, 最大值={df['pred_in'].max():.2f}")
print(f"  流出: 最小值={df['pred_out'].min():.2f}, 最大值={df['pred_out'].max():.2f}")
print(f"  Betweenness: 最小值={df['pred_between'].min():.2f}, 最大值={df['pred_between'].max():.2f}")

print(f"\n真实值统计:")
print(f"  流入: 最小值={df['PLAY_D_IN'].min():.2f}, 最大值={df['PLAY_D_IN'].max():.2f}")
print(f"  流出: 最小值={df['PLAY_O_OUT'].min():.2f}, 最大值={df['PLAY_O_OUT'].max():.2f}")
print(f"  Betweenness: 最小值={df['PLAY__Betweenness'].min():.2f}, 最大值={df['PLAY__Betweenness'].max():.2f}")

# =========================
# 9. 参数/特征重要性与可解释性分析（RF + SHAP + 权重分析）
# =========================

# 输出目录
_output_dir = "regession/output"
os.makedirs(_output_dir, exist_ok=True)

_shape_features = [
   "building_ratio0", "railway_ratio1", "road_ratio2", "water_ratio3",
    "instru_ratio4", "green_ratio5", "square_ratio6", "undevelop_ratio7",
    "playground_ratio8", "LPI_Max", "PR_COUNTclass", "SHI_zhouchang",
    "RND_road", "Social", "Life", "Entertainment",
    "Production", "POI_density", "MAX_Elevation", "MEAN_Elevation",
]
_target_cols = ["PLAY_D_IN", "PLAY_O_OUT", "PLAY__Betweenness"]

def analyze_feature_importance_rf(X_np, Y_np):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, target_name in enumerate(_target_cols):
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_np, Y_np[:, i])
        importance = rf.feature_importances_
        indices = np.argsort(importance)[::-1]
        axes[i].bar(range(len(importance)), importance[indices])
        axes[i].set_xticks(range(len(importance)))
        axes[i].set_xticklabels([_shape_features[j] for j in indices], rotation=45, ha='right')
        axes[i].set_title(f'{target_name} - 特征重要性(RF)')
        axes[i].set_ylabel('重要性')
        # 保存CSV
        df_imp = pd.DataFrame({
            'feature': _shape_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        df_imp.to_csv(os.path.join(_output_dir, f'feature_importance_rf_{target_name}.csv'), index=False)
    plt.tight_layout()
    plt.savefig(os.path.join(_output_dir, 'feature_importance_rf.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_shap_rf(X_np, Y_np):
    if not _shap_available:
        print('⚠️ shap 未安装，跳过 SHAP 分析。可运行: pip install shap')
        return
    for i, target_name in enumerate(_target_cols):
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_np, Y_np[:, i])
        explainer = shap.TreeExplainer(rf)
        sample_size = min(300, len(X_np))
        sample_idx = np.random.choice(len(X_np), sample_size, replace=False)
        X_sample = X_np[sample_idx]
        shap_values = explainer.shap_values(X_sample)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=_shape_features, show=False, plot_type='bar')
        plt.title(f'{target_name} - SHAP特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(_output_dir, f'shap_summary_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def analyze_model_weights_gnn(model_inst):
    try:
        conv1_weights = model_inst.conv1.weight.data.numpy()
    except Exception:
        print('⚠️ 无法访问第一层GCN权重，跳过权重分析。')
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(conv1_weights.flatten(), bins=60, alpha=0.8, color='steelblue')
    axes[0].set_title('第一层GCN权重分布')
    axes[0].set_xlabel('权重值')
    axes[0].set_ylabel('频次')
    feature_importance = np.mean(np.abs(conv1_weights), axis=0)
    idx = np.argsort(feature_importance)[::-1]
    axes[1].bar(range(len(feature_importance)), feature_importance[idx])
    axes[1].set_xticks(range(len(feature_importance)))
    axes[1].set_xticklabels([_shape_features[j] for j in idx], rotation=45, ha='right')
    axes[1].set_title('基于第一层权重的特征重要性')
    axes[1].set_ylabel('平均绝对权重')
    plt.tight_layout()
    plt.savefig(os.path.join(_output_dir, 'gnn_weight_based_importance_play.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 保存数值
    pd.DataFrame({
        'feature': _shape_features,
        'abs_weight_mean': feature_importance
    }).sort_values('abs_weight_mean', ascending=False).to_csv(
        os.path.join(_output_dir, 'gnn_weight_based_importance_play.csv'), index=False
    )

# =========================
# 10. 每个网格的特征重要性分析（新增功能）
# =========================
def analyze_per_grid_feature_importance(X_np, Y_np, coords, df_original):
    """
    计算每个网格的每个特征重要性，并输出到CSV
    """
    print('\n开始计算每个网格的特征重要性...')
    
    # 创建输出目录
    grid_importance_dir = os.path.join(_output_dir, 'grid_feature_importance')
    os.makedirs(grid_importance_dir, exist_ok=True)
    
    # 为每个目标变量计算特征重要性
    for i, target_name in enumerate(_target_cols):
        print(f'正在计算 {target_name} 的每个网格特征重要性...')
        
        # 使用随机森林计算SHAP值
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_np, Y_np[:, i])
        
        # 计算SHAP值
        if _shap_available:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_np)
            
            # 创建每个网格的特征重要性DataFrame
            grid_importance_data = []
            
            for grid_idx in range(len(X_np)):
                grid_data = {
                    'grid_id': df_original.iloc[grid_idx]['id2019'],  # 使用原始数据中的id2019
                    'lon': coords[grid_idx, 0],
                    'lat': coords[grid_idx, 1],
                    'actual_value': Y_np[grid_idx, i],
                    'predicted_value': rf.predict(X_np[grid_idx:grid_idx+1])[0]
                }
                
                # 添加每个特征的SHAP值（特征重要性）
                for feature_idx, feature_name in enumerate(_shape_features):
                    grid_data[f'{feature_name}_shap_value'] = shap_values[grid_idx, feature_idx]
                    grid_data[f'{feature_name}_feature_value'] = X_np[grid_idx, feature_idx]
                
                grid_importance_data.append(grid_data)
            
            # 转换为DataFrame并保存
            grid_df = pd.DataFrame(grid_importance_data)
            output_file = os.path.join(grid_importance_dir, f'grid_feature_importance_{target_name}.csv')
            grid_df.to_csv(output_file, index=False)
            
            print(f'✅ {target_name} 的每个网格特征重要性已保存到: {output_file}')
            
            # 创建特征重要性汇总统计
            summary_data = []
            for feature_name in _shape_features:
                shap_col = f'{feature_name}_shap_value'
                summary_data.append({
                    'feature_name': feature_name,
                    'mean_shap_value': grid_df[shap_col].mean(),
                    'std_shap_value': grid_df[shap_col].std(),
                    'abs_mean_shap_value': grid_df[shap_col].abs().mean(),
                    'max_shap_value': grid_df[shap_col].max(),
                    'min_shap_value': grid_df[shap_col].min(),
                    'feature_importance_rank': grid_df[shap_col].abs().mean()
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('feature_importance_rank', ascending=False)
            summary_file = os.path.join(grid_importance_dir, f'feature_importance_summary_{target_name}.csv')
            summary_df.to_csv(summary_file, index=False)
            
            print(f'✅ {target_name} 的特征重要性汇总已保存到: {summary_file}')
            
        else:
            print(f'⚠️ SHAP 未安装，无法计算 {target_name} 的每个网格特征重要性')
    
    # 创建所有网格的综合特征重要性表
    if _shap_available:
        print('\n创建综合特征重要性表...')
        
        # 合并所有目标变量的特征重要性
        all_features_data = []
        
        for grid_idx in range(len(X_np)):
            grid_data = {
                'grid_id': df_original.iloc[grid_idx]['id2019'],  # 使用原始数据中的id2019
                'lon': coords[grid_idx, 0],
                'lat': coords[grid_idx, 1]
            }
            
            # 添加每个目标变量的实际值和预测值
            for i, target_name in enumerate(_target_cols):
                grid_data[f'{target_name}_actual'] = Y_np[grid_idx, i]
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X_np, Y_np[:, i])
                grid_data[f'{target_name}_predicted'] = rf.predict(X_np[grid_idx:grid_idx+1])[0]
            
            # 添加每个特征的原始值
            for feature_idx, feature_name in enumerate(_shape_features):
                grid_data[f'{feature_name}_value'] = X_np[grid_idx, feature_idx]
            
            # 添加每个特征对每个目标变量的SHAP值
            for i, target_name in enumerate(_target_cols):
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                rf.fit(X_np, Y_np[:, i])
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_np[grid_idx:grid_idx+1])
                
                for feature_idx, feature_name in enumerate(_shape_features):
                    grid_data[f'{feature_name}_shap_{target_name}'] = shap_values[0, feature_idx]
            
            all_features_data.append(grid_data)
        
        # 保存综合表
        comprehensive_df = pd.DataFrame(all_features_data)
        comprehensive_file = os.path.join(grid_importance_dir, 'comprehensive_grid_feature_importance.csv')
        comprehensive_df.to_csv(comprehensive_file, index=False)
        
        print(f'✅ 综合特征重要性表已保存到: {comprehensive_file}')
        
        # 创建特征重要性热力图数据
        print('\n创建特征重要性热力图数据...')
        
        for i, target_name in enumerate(_target_cols):
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_np, Y_np[:, i])
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_np)
            
            # 创建热力图数据
            heatmap_data = []
            for grid_idx in range(len(X_np)):
                for feature_idx, feature_name in enumerate(_shape_features):
                    heatmap_data.append({
                        'grid_id': df_original.iloc[grid_idx]['id2019'],  # 使用原始数据中的id2019
                        'lon': coords[grid_idx, 0],
                        'lat': coords[grid_idx, 1],
                        'feature_name': feature_name,
                        'shap_value': shap_values[grid_idx, feature_idx],
                        'abs_shap_value': abs(shap_values[grid_idx, feature_idx]),
                        'feature_value': X_np[grid_idx, feature_idx]
                    })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_file = os.path.join(grid_importance_dir, f'feature_importance_heatmap_data_{target_name}.csv')
            heatmap_df.to_csv(heatmap_file, index=False)
            
            print(f'✅ {target_name} 的特征重要性热力图数据已保存到: {heatmap_file}')

# 执行分析
print('\n开始参数/特征重要性与可解释性分析...')
X_np = X_processed  # 与训练时一致
Y_np = Y_original
analyze_feature_importance_rf(X_np, Y_np)
analyze_shap_rf(X_np, Y_np)
analyze_model_weights_gnn(model)

# 执行每个网格的特征重要性分析
analyze_per_grid_feature_importance(X_np, Y_np, coords, df)

print('✅ 分析完成。结果已保存至 regession/output/ 目录。')
