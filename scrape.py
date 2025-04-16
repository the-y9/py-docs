#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://ml.gatech.edu/node/2271'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')

# Find the table in the HTML
table = soup.find('table')
#%%
data = {}
rows = table.find_all('tr')
# Iterate through each row in the table
headers = [th.text.strip() for th in rows[0].find_all('th')]
#%%
for idx, row in enumerate(rows[1:]):
    cols = [td.text.strip() for td in row.find_all('td')]
    raw = {headers[i]: cols[i] for i in range(len(headers))}
    data[idx+1] = raw

#%%
df = pd.DataFrame.from_dict(data, orient='index', columns=headers)
df
# %%
