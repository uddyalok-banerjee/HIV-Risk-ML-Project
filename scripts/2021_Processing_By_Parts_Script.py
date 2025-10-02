# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:59:50 2023

@author: ubanerje
"""

import os
import pandas as pd

# Directory path
directory = 'C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/UnseenTest6_Try2/'

# Get the file names in the directory
file_names = os.listdir(directory)

# Create a dataframe with the file names
df = pd.DataFrame({'File Name': file_names})
df['File Name'] = df['File Name'].apply(lambda x: os.path.splitext(x)[0])
#df['File Name'] = df['File Name'].str.strip('"\'')
processed_filenames = df['File Name'].tolist()

filtered_data = data[~data['HSP_Account_ID'].isin(processed_filenames)]
filtered_data.to_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/2021_CUIS_2_Part4_Final_DeDup_2.csv', index=False)