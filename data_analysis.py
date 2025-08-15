import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
filename = "data/preprocessed_data.csv"

if not os.path.exists(filename):
    print(f"❌ 找不到檔案：{filename}")
    exit()

# 讀入 CSV 檔
df_all = pd.read_csv(filename)

# 確認是否有 NaN
nan_exists = df_all.isnull().any().any()
if nan_exists:
    print("⚠️ 資料中仍存在 NaN：")
    print(df_all.isnull().sum())
else:
    print("✅ 無 NaN 值")

# 統計摘要
print("\n📊 數值統計摘要：")
print(df_all.describe())

# 自動判斷特徵數（排除 death 欄位）
feature_cols = [col for col in df_all.columns if col != 'death' and col != 'subject_id']
n_features = len(feature_cols)

# 畫出每個特徵的分布（依 death 分組）
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols
plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for i, col in enumerate(feature_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.kdeplot(data=df_all, x=col, hue='death', fill=True, common_norm=False, alpha=0.5)
    plt.title(f'Feature {col} (by death)')
    plt.xlabel("Value")
    plt.ylabel("Density")

plt.tight_layout()
plt.savefig("feature_by_death.png")
print("\n📈 分組圖已儲存為 feature_by_death.png")
