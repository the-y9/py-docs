#%% ml learning
import numpy as np
import pandas as pd

# Load data
try:
    data = pd.read_excel('ml_data.xlsx')
    print(data.head())
except Exception as e:
    print(e)

#%%
# Create frequency table for class wise
if 'Weather' in data.columns:
    frequency_table = data['Weather'].value_counts()
    print(frequency_table)
else:
    print("Column 'class' not found in the data.")
