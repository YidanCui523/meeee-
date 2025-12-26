import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import xgboost as xgb

def evaluate_model(name, model, X_train, X_test, y_train, y_test, param_grid=None, poly_degree=None, file_base=None):
    if poly_degree:
        X_train_poly = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(X_train)
        X_test_poly = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(X_test)
    else:
        X_train_poly, X_test_poly = X_train, X_test

    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid.fit(X_train_poly, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_scores = grid.cv_results_['mean_test_score']
    else:
        best_model = model.fit(X_train_poly, y_train)
        best_params = 'N/A'
        cv_scores = 'N/A'

    y_pred = best_model.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # 学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train_poly, y_train, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='训练集分数')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='验证集分数')
    plt.xlabel('训练样本数')
    plt.ylabel('R² 分数')
    plt.title(f'学习曲线 - {name}')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'result/LearningCurve_{file_base}_{name}_work.png')
    plt.close()

    # 保存报告
    txt_path = f'result/Report_{file_base}_{name}.txt'
    with open(txt_path, 'w',encoding='utf-8') as f:
        f.write(f"模型: {name}\n")
        f.write(f"最佳参数: {best_params}\n")
        f.write(f"测试集 R²: {r2:.4f}\n")
        f.write(f"测试集 MAE: {mae:.4f}\n")
        f.write(f"测试集 MSE: {mse:.4f}\n")
        f.write(f"测试集 RMSE: {rmse:.4f}\n")
        f.write(f"交叉验证分数: {cv_scores}\n")
    print(f"{name} 报告已保存到 {txt_path}")

    return {'Model': name, 'BestParams': best_params, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

def main(file_path):
    os.makedirs('result', exist_ok=True)
    file_base = os.path.basename(file_path).replace('.csv', '')

    # 读取数据
    df = pd.read_csv(file_path)
    
    # 定义自变量（特征）
    feature_columns = [
        "building_ratio0", "railway_ratio1", "road_ratio2", "water_ratio3",
        "instru_ratio4", "green_ratio5", "square_ratio6", "undevelop_ratio7",
        "playground_ratio8", "LPI_Max", "PR_COUNTclass", "SHI_zhouchang",
        "RND_road", "Social", "Life", "Entertainment",
        "Production", "POI_density", "MAX_Elevation", "MEAN_Elevation"
    ]
    
    # 定义因变量（目标变量）
    target_columns = ["WORK_D_IN", "WORK_O_OUT", "WORK__Betweenness"]
    
    # 提取特征和目标变量
    X = df[feature_columns].values
    
    # 存储所有结果
    all_results = []
    
    # 对每个目标变量分别进行建模
    for target_name in target_columns:
        print(f"\n=== 开始建模 {target_name} ===")
        
        y = df[target_name].values
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = []

        # OLS
        print(f"训练 OLS 模型...")
        results.append(evaluate_model(
            'OLS',
            LinearRegression(),
            X_train, X_test, y_train, y_test,
            file_base=f"{file_base}_{target_name}"
        ))

        # Random Forest
        print(f"训练 Random Forest 模型...")
        rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        results.append(evaluate_model(
            'RandomForest',
            RandomForestRegressor(random_state=42),
            X_train, X_test, y_train, y_test,
            param_grid=rf_params,
            file_base=f"{file_base}_{target_name}"
        ))

        # XGBoost
        print(f"训练 XGBoost 模型...")
        xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
        results.append(evaluate_model(
            'XGBoost',
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            X_train, X_test, y_train, y_test,
            param_grid=xgb_params,
            file_base=f"{file_base}_{target_name}"
        ))

        # Polynomial Ridge
        print(f"训练 Polynomial Ridge 模型...")
        ridge_params = {'alpha': np.logspace(-3, 3, 5)}
        results.append(evaluate_model(
            'PolynomialRidge',
            Ridge(),
            X_train, X_test, y_train, y_test,
            param_grid=ridge_params,
            poly_degree=2,
            file_base=f"{file_base}_{target_name}"
        ))

        # MLP
        print(f"训练 MLP 模型...")
        mlp_params = {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
        results.append(evaluate_model(
            'MLP',
            MLPRegressor(max_iter=1000, random_state=42),
            X_train, X_test, y_train, y_test,
            param_grid=mlp_params,
            file_base=f"{file_base}_{target_name}"
        ))

        # 为每个目标变量添加目标变量名称
        for result in results:
            result['Target'] = target_name
        
        # 保存单个目标变量的结果
        results_df = pd.DataFrame(results)
        summary_path = f'result/Model_Comparison_{file_base}_{target_name}_work.csv'
        results_df.to_csv(summary_path, index=False)
        print(f"{target_name} 的模型对比结果已保存到 {summary_path}")
        
        # 添加到总结果中
        all_results.extend(results)

    # 汇总所有结果
    all_results_df = pd.DataFrame(all_results)
    summary_path = f'result/Model_Comparison_{file_base}_All_Targets_work.csv'
    all_results_df.to_csv(summary_path, index=False)
    print(f"\n所有目标变量的模型对比结果已保存到 {summary_path}")
    
    # 创建按模型分组的汇总表
    model_summary = []
    for model_name in ['OLS', 'RandomForest', 'XGBoost', 'PolynomialRidge', 'MLP']:
        model_results = all_results_df[all_results_df['Model'] == model_name]
        model_summary.append({
            'Model': model_name,
            'WORK_D_IN_R2': model_results[model_results['Target'] == 'WORK_D_IN']['R2'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_D_IN']) > 0 else None,
            'WORK_D_IN_MAE': model_results[model_results['Target'] == 'WORK_D_IN']['MAE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_D_IN']) > 0 else None,
            'WORK_D_IN_RMSE': model_results[model_results['Target'] == 'WORK_D_IN']['RMSE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_D_IN']) > 0 else None,
            'WORK_O_OUT_R2': model_results[model_results['Target'] == 'WORK_O_OUT']['R2'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_O_OUT']) > 0 else None,
            'WORK_O_OUT_MAE': model_results[model_results['Target'] == 'WORK_O_OUT']['MAE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_O_OUT']) > 0 else None,
            'WORK_O_OUT_RMSE': model_results[model_results['Target'] == 'WORK_O_OUT']['RMSE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK_O_OUT']) > 0 else None,
            'WORK__Betweenness_R2': model_results[model_results['Target'] == 'WORK__Betweenness']['R2'].iloc[0] if len(model_results[model_results['Target'] == 'WORK__Betweenness']) > 0 else None,
            'WORK__Betweenness_MAE': model_results[model_results['Target'] == 'WORK__Betweenness']['MAE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK__Betweenness']) > 0 else None,
            'WORK__Betweenness_RMSE': model_results[model_results['Target'] == 'WORK__Betweenness']['RMSE'].iloc[0] if len(model_results[model_results['Target'] == 'WORK__Betweenness']) > 0 else None,
        })
    
    model_summary_df = pd.DataFrame(model_summary)
    model_summary_path = f'result/Model_Summary_{file_base}_work.csv'
    model_summary_df.to_csv(model_summary_path, index=False)
    print(f"模型汇总表已保存到 {model_summary_path}")
    
    # 打印结果摘要
    print("\n=== 模型性能摘要 ===")
    print("\n按目标变量分组的最佳模型:")
    for target in target_columns:
        target_results = all_results_df[all_results_df['Target'] == target]
        best_model = target_results.loc[target_results['R2'].idxmax()]
        print(f"{target}: {best_model['Model']} (R² = {best_model['R2']:.4f})")
    
    print("\n按模型分组的平均性能:")
    for model_name in ['OLS', 'RandomForest', 'XGBoost', 'PolynomialRidge', 'MLP']:
        model_results = all_results_df[all_results_df['Model'] == model_name]
        avg_r2 = model_results['R2'].mean()
        avg_mae = model_results['MAE'].mean()
        avg_rmse = model_results['RMSE'].mean()
        print(f"{model_name}: 平均R² = {avg_r2:.4f}, 平均MAE = {avg_mae:.4f}, 平均RMSE = {avg_rmse:.4f}")

if __name__ == '__main__':
    # 输入文件名（替换这里即可）
    main('regession/work2019_0906.csv')
