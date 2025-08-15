import pandas as pd
import numpy as np
from tqdm import tqdm

# 設定參數
TIME_LIMIT_HOURS = 48  # 只取入院後 48 小時內的資料
TIME_BIN_HOURS = 1     # 每小時為一格
N_BINS = TIME_LIMIT_HOURS // TIME_BIN_HOURS

# 讀取資料
labevents = pd.read_csv("data/hosp/_labevents.csv")
chartevents = pd.read_csv("data/icu/_chartevents.csv")
patients = pd.read_csv("data/hosp/_patients.csv")
admissions = pd.read_csv("data/hosp/_admissions.csv")
label_death = pd.read_csv("data/label/_label_death.csv")

# 定義重要 itemid 對應
lab_itemids = {
    51006: 'BUN', 50863: 'Alkaline Phosphatase', 50885: 'Bilirubin',
    50912: 'Creatinine', 50931: 'Glucose', 51265: 'Platelets', 51222: 'Hemoglobin'
}
chart_itemids = {
    220045: 'Heart Rate', 220210: 'Respiratory Rate', 220052: 'Mean Arterial Pressure',
    223762: 'Temperature', 220179: 'Systolic Blood Pressure',
    224639: 'Weight',  226730: 'Height'
}
important_itemids = {**lab_itemids, **chart_itemids}  # 合併所有

# 篩選資料
labevents_filtered = labevents[labevents['itemid'].isin(lab_itemids.keys())].copy()
chartevents_filtered = chartevents[chartevents['itemid'].isin(chart_itemids.keys())].copy()
labevents_filtered['source'] = 'lab'
chartevents_filtered['source'] = 'chart'
combined = pd.concat([labevents_filtered, chartevents_filtered])

# 時間處理與合併 patient/admission info
combined['charttime'] = pd.to_datetime(combined['charttime'], errors='coerce')
patients['gender'] = patients['gender'].fillna(-1)
combined = combined.merge(patients[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')
admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
combined = combined.merge(admissions[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'], how='left')
combined['hours_from_admit'] = (combined['charttime'] - combined['admittime']).dt.total_seconds() / 3600
combined = combined[combined['hours_from_admit'] >= 0]
combined = combined[combined['hours_from_admit'] <= TIME_LIMIT_HOURS]

# 加入死亡標籤
combined['death'] = 0
death_index = label_death.set_index(['subject_id', 'hadm_id']).index
combined.loc[combined.set_index(['subject_id', 'hadm_id']).index.isin(death_index), 'death'] = 1

# 數值處理
combined['valuenum'] = pd.to_numeric(combined['valuenum'], errors='coerce')

# 建立 [T, F] 資料，每筆為一組 hadm_id
tensor_dict = {}
label_dict = {}

feature_list = list(important_itemids.keys())
grouped = combined.groupby(['subject_id', 'hadm_id'])

for (sid, hid), group in tqdm(grouped):
    matrix = np.full((N_BINS, len(feature_list)), np.nan)
    time_bins = np.arange(0, TIME_LIMIT_HOURS + TIME_BIN_HOURS, TIME_BIN_HOURS)

    for _, row in group.iterrows():
        bin_index = int(row['hours_from_admit'] // TIME_BIN_HOURS)
        if bin_index >= N_BINS:
            continue
        try:
            f_index = feature_list.index(row['itemid'])
            matrix[bin_index][f_index] = row['valuenum']
        except:
            continue

    df_matrix = pd.DataFrame(matrix, columns=feature_list)

    # ====== 缺值處理階段 1（每個 group 內）======
    df_matrix = df_matrix.interpolate(limit_direction='both')  # 插值法
    df_matrix = df_matrix.ffill().bfill()  # LOCF + NOCB

    # ====== 缺值處理階段 2（整體中位數填補）======
    for f in feature_list:
        if df_matrix[f].isnull().any():
            df_matrix[f] = df_matrix[f].fillna(combined[combined['itemid'] == f]['valuenum'].median())

    tensor_dict[(sid, hid)] = df_matrix.values
    label_dict[(sid, hid)] = group['death'].iloc[0]

# 類別變量處理（gender）
for k in label_dict.keys():
    if isinstance(patients[patients['subject_id'] == k[0]]['gender'].values, np.ndarray):
        g = patients[patients['subject_id'] == k[0]]['gender'].values
        label_dict[k] = (label_dict[k], g[0] if len(g) > 0 else -1)
    else:
        label_dict[k] = (label_dict[k], -1)

# 儲存結果為 npz 檔
np.savez("timeseries_input_label.npz", inputs=tensor_dict, labels=label_dict)
print("✅ 預處理完成，已儲存為 timeseries_input_label.npz")
