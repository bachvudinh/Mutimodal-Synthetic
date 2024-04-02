import pandas as pd
import json
import os
import numpy as np
import ast
path = 'mcocr_train_df_rotate_filtered.csv'
data_csv = pd.read_csv(path)
# create a json to store the data
path_save = 'mcocr_train_df_rotate_filtered.json'
data_new = []
# iterate through the rows of the dataframe and print the first 5 rows

for index, row in data_csv.iterrows():
    texts = row['anno_texts'].split("|||")
    labels = row['anno_labels'].split("|||")
    mc_ocr_text = []
    temp_address=["Địa chỉ xuất hóa đơn là: "]
    temp_seller = ["Đơn vị bán là: "]
    temp_timestamp = ["Thời gian xuất hóa đơn là: "]
    temp_total_cost = ["Tổng tiền phải thanh toán: "]
    for text, label in zip(texts, labels):
        if label == 'ADDRESS':
            if text != "":
                if len(temp_address) == 1:
                    temp_address.append(text)
                else:
                    temp_address.append(', ' + text)
        elif label == 'SELLER':
            if text != "":
                if len(temp_seller) == 1:
                    temp_seller.append(text)
                else:
                    temp_seller.append(', ' + text)
        elif label == 'TIMESTAMP':
            if text != "":
                if len(temp_timestamp) == 1:
                    temp_timestamp.append(text)
                else:
                    temp_timestamp.append(', ' + text)
        elif label == 'TOTAL_COST':  
            if text != "":
                if len(temp_total_cost) == 1:
                    temp_total_cost.append(text.replace(":", "")) 
                elif len(temp_total_cost) == 2:
                    temp_total_cost.append(" là ") 
                    temp_total_cost.append(text.replace(":", "")) 
    if len(temp_seller) > 1:
        mc_ocr_text.append(''.join(temp_seller))
    if len(temp_address) > 1:
        mc_ocr_text.append(''.join(temp_address))
    if len(temp_timestamp) > 1:
        mc_ocr_text.append(''.join(temp_timestamp))
    if len(temp_total_cost) > 1:
        mc_ocr_text.append(''.join(temp_total_cost))
    data_new.append({
        'image_id': row['img_id'],
        'texts': mc_ocr_text
    })
# save to path_save, ensure ascii encoding = False
with open(path_save, 'w') as outfile:
    json.dump(data_new, outfile, ensure_ascii=False, indent=4)