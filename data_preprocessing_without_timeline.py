import numpy as np
import pandas as pd
from data_functions import read_data

# read data
chart_events_filtered_data = read_data('data/icu/_chartevents_filtered.csv')
label_events_filtered_data = read_data('data/icu/_labevents_filtered.csv')
patient_basic = read_data("data/hosp/_patients.csv")
death = read_data("data/label/_label_death.csv")

# convert charttime to sec
chart_events_filtered_data['charttime(sec)'] = pd.to_datetime(chart_events_filtered_data['charttime']).apply(lambda x: int(x.timestamp()))
label_events_filtered_data['charttime(sec)'] = pd.to_datetime(label_events_filtered_data['charttime']).apply(lambda x: int(x.timestamp()))


# Get the Average value of each itemid
chart_events_groups = chart_events_filtered_data.groupby(['subject_id', "itemid"])
dict = {"subject_id":[],
        "itemid":[],
        "value": [],}

for (a, b), group in chart_events_groups:
    dict["subject_id"].append(group["subject_id"].iloc[0])
    dict["itemid"].append(group["itemid"].iloc[0])
    dict["value"].append(sum(group["value"]) / len(group["value"]))
grouped_chart_events_filtered_data = pd.DataFrame(dict)

def get(values): # to get true value
    x = 0
    l = 0
    for v in values:
        if v != '___':
            x += float(v)
            l += 1
    if l ==0:
        return 0
    return x/l

label_events_groups = label_events_filtered_data.groupby(['subject_id', "itemid"])
dict = {"subject_id":[],
        "itemid":[],
        "value": [],}

for (a, b), group in label_events_groups:
    dict["subject_id"].append(group["subject_id"].iloc[0])
    dict["itemid"].append(group["itemid"].iloc[0])
    dict["value"].append(get(group["value"]))
grouped_label_events_filtered_data = pd.DataFrame(dict)


# Merge Data
chart_events_wide = grouped_chart_events_filtered_data.pivot(index='subject_id', columns='itemid', values='value')
label_event_wide = grouped_label_events_filtered_data.pivot(index='subject_id', columns='itemid', values='value')
patiens_data = pd.merge(label_event_wide, chart_events_wide, on='subject_id', how='outer')
patiens_data = pd.merge(patiens_data, patient_basic[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')


# Encoding Categorical Data
patiens_data["gender"] = patiens_data["gender"].apply(lambda x: 0 if x == 'M' else 1) 

# Fill Missing Data
for col in patiens_data.columns:
    if patiens_data[col].dtype in ['float64', 'int64']:  
        median_value = patiens_data[col].median()
        patiens_data[col] = patiens_data[col].fillna(median_value)
        
# Standardize Numeric Data
for col in patiens_data.columns:
    if col == "subject_id" or col == "gender": continue
    if patiens_data[col].dtype in ['float64', 'int64']:  # 只處理數值欄位
        mu = np.mean( patiens_data[col])
        std = np.std( patiens_data[col])
        patiens_data[col] = patiens_data[col].apply(lambda x: (x - mu)/std)
        
# Merge with label
death_id = death["subject_id"]
patiens_data["death"] = patiens_data["subject_id"].apply(lambda x: 1 if x in death_id.values else 0)

patiens_data.to_csv("data/preprocessed_data.csv", index=False)
