# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:54:41 2023

@author: ubanerje
"""

import os
import pandas as pd
import sqlite3
import pyodbc

#Run SQL query to create table - load in dataframe
cnxn_str = ("Driver={SQL Server Native Client 11.0};"
            "Server=clarityprod.rush.edu;"
            "Database=Rush_Extract;"
            "Trusted_Connection=yes;")

cnxn = pyodbc.connect(cnxn_str)
with open('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/queryrebuild.sql', 'r') as f:
    query = f.read()
data_sql = pd.read_sql(query, cnxn)
cnxn.close()

#Read CSV 2020 Results
data = pd.read_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/2022_CUIS_p1.csv', sep = '|')
condensed_data2 = data.groupby('HSP_ACCOUNT_ID')['CUIS'].apply(lambda x: [''.join(vals) for vals in x]).reset_index() 

print(condensed_data2)
print(condensed_data2.shape[0])



#data3.to_excel('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/data3.xlsx', index=False)
data_sql.to_csv('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/rebuild_CUIS.csv', sep = '|', index=False)


#data = pd.read_excel('C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Oct22CUIS.xlsx')
data = data_sql
data = data.astype({'HSP_ACCOUNT_ID':'string'})

# Define the path to the folder where you want to save the text files
folder_path = 'C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/Rebuild'

# Loop through each unique value in the column
for value in data['CUIS'].unique():

    # Create a new text file with the value from the other column as the filename
    filename = os.path.join(folder_path, data.loc[data['CUIS'] == value, 'HSP_ACCOUNT_ID'].iloc[0] + '.txt')
    with open(filename, "w") as file:

        # Write each row of the data where the column contains the value to the file
        for index, row in data.iterrows():
            if row['CUIS'] == value:
                file.write(f"{row['CUIS']}")
