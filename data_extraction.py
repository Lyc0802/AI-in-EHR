from data_functions import read_data


chart_event = read_data('data/icu/_chartevents.csv')
label_event = read_data("data/hosp/_labevents.csv")
patient_basic = read_data("data/hosp/_patients.csv")
death = read_data("data/label/_label_death.csv")


# Extract Important Data from chart_event and label_event

chart_event_fitemIds = [220045, 220210, 220052, 223762, 220179, 226730, 224639]
chart_event_filtered_data = chart_event[chart_event['itemid'].isin(chart_event_fitemIds)]
# store as csv
chart_event_filtered_data.to_csv('data/icu/_chartevents_filtered.csv', index=False)

label_event_itemIds = [51006, 50863, 50885, 50912, 50931, 51265, 51222]
label_event_filtered_data = label_event[label_event['itemid'].isin(label_event_itemIds)]
# store as csv
label_event_filtered_data.to_csv('data/icu/_labevents_filtered.csv', index=False)