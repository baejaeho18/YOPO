import pandas as pd
import os
import csv
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse
import html
import joblib

# Load CSV file into a DataFrame
df_orig = pd.read_csv("/yopo-artifact/data/dataset/from_pagegraph/features_raw.csv")
df = pd.read_csv("/yopo-artifact/data/dataset/from_pagegraph/features_raw.csv")

# Label encoding categorical features and save info
label_encoder = LabelEncoder()
columns_to_encode = ['FEATURE_RESOURCE_TYPE']
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

column = "FEATURE_RESOURCE_TYPE"
joblib.dump(label_encoder, f'/yopo-artifact/scripts/crawler/pagegraph/encoding_pagegraph/{column}_encoder.joblib')

# change AD to 1, NonAD to 0
df.dropna(subset=['CLASS'], inplace=True)
df['CLASS'] = df['CLASS'].replace({"AD": 1, "NONAD":0})
df['FEATURE_FROM_SUBDOMAIN'] = df['FEATURE_FROM_SUBDOMAIN'].replace({True: int(1), False: int(0)})
df['FEATURE_FROM_THIRD_PARTY'] = df['FEATURE_FROM_THIRD_PARTY'].replace({True: int(1), False: int(0)})
df['FEATURE_SEMICOLON_IN_QUERY'] = df['FEATURE_SEMICOLON_IN_QUERY'].replace({True: int(1), False: int(0)})
df['FEATURE_MODIFIED_BY_SCRIPT'] = df['FEATURE_MODIFIED_BY_SCRIPT'].replace({True: int(1), False: int(0)})
df['FEATURE_PARENT_MODIFIED_BY_SCRIPT'] = df['FEATURE_PARENT_MODIFIED_BY_SCRIPT'].replace({True: int(1), False: int(0)})

df_orig['FEATURE_FROM_SUBDOMAIN'] = df_orig['FEATURE_FROM_SUBDOMAIN'].replace({True: int(1), False: int(0)})
df_orig['FEATURE_FROM_THIRD_PARTY'] = df_orig['FEATURE_FROM_THIRD_PARTY'].replace({True: int(1), False: int(0)})
df_orig['FEATURE_SEMICOLON_IN_QUERY'] = df_orig['FEATURE_SEMICOLON_IN_QUERY'].replace({True: int(1), False: int(0)})
df_orig['FEATURE_MODIFIED_BY_SCRIPT'] = df_orig['FEATURE_MODIFIED_BY_SCRIPT'].replace({True: int(1), False: int(0)})
df_orig['FEATURE_PARENT_MODIFIED_BY_SCRIPT'] = df_orig['FEATURE_PARENT_MODIFIED_BY_SCRIPT'].replace({True: int(1), False: int(0)})

# save original one
df.to_csv("/yopo-artifact/data/dataset/from_pagegraph/features_rf_all.csv", index=False)

target_rows = []
unique_urls = []


def check_string_in_html_file(file_path, target_string):
    try:
        with open(file_path, 'r') as file:
            try:
                html_content = file.read()
            except:
                # print("READING HTML ERROR!")
                # print(file_path)
                return False
            decoded_html = html.unescape(html_content)
            if f'"{target_string}"' in decoded_html:
                return True
            elif f"'{target_string}'" in decoded_html:
                return True
            else:
                return False
    except:
        return False


def create_dictionary_final_url_mapping(csv_file):
    result_dict = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            result_dict[row[0]] = row[2]
    return result_dict


mapping_path = '/yopo-artifact/data/rendering_stream/final_url_to_html_filepath_mapping.csv'
mapping_dict_final_url_mapping = create_dictionary_final_url_mapping(mapping_path)
log_history = set()

# We sample 2,400 target requests
# We only need 2,000 target requests; however, some requests may not work when processed with the bs4 library.
# Therefore, we oversample the target requests and remove unnecessary ones in the later steps.
while len(target_rows) < 2400:
    # Progress..
    if len(target_rows) % 100 == 0 and len(target_rows) not in log_history:
        print("Now gathering {} targets...".format(len(target_rows)))
        log_history.add(len(target_rows))
    
    # Extract values
    selected_row = df.sample(n=1).iloc[0]
    
    url = selected_row['FINAL_URL']
    requested_url = selected_row['NETWORK_REQUEST_URL']
    try:
        html_fpath = mapping_dict_final_url_mapping[selected_row['FINAL_URL']]
    except:
        continue
    html_file = html_fpath.split("/html/")[-1]
    is_AD = selected_row['CLASS']
    
    # check AD
    if is_AD == "NONAD":
        continue
    
    if url == requested_url:
        # print("skip same url")
        pass
    
    # check requested url inside the html file.
    if check_string_in_html_file(html_fpath, requested_url):
        pass
        # print("TRUE")
        # print(requested_url, html_file)
    else:
        continue
        # print("FALSE")
        # print(requested_url, html_file)
    
    # check duplicates of target_rows
    if requested_url in unique_urls:
        continue
    unique_urls.append(requested_url)
    
    # Add to the target row
    target_rows.append(selected_row)

# Transpose to DataFrame
target_df = pd.concat(target_rows, axis=1).transpose()
selected_indices = target_df.index
df = df.drop(selected_indices)
df_orig = df_orig.drop(selected_indices)

# Save the encoded DataFrame to a new CSV file
df.to_csv("/yopo-artifact/data/dataset/from_pagegraph/features_rf.csv", index=False)
df_orig.to_csv("/yopo-artifact/data/dataset/from_pagegraph/features_raw_except_target.csv", index=False)
target_df.to_csv("/yopo-artifact/data/dataset/from_pagegraph/target_features_rf_oversampled.csv", index=False)
