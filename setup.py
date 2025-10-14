# download pdf files 
import pandas as pd 
import requests
import shutil
import os

df=pd.read_excel('data/raw/Medical_reports2.xlsx') 
file_urls=df['URL'] 
for url in file_urls: 
    # download the pdf from the url 
    filename=url.split('/')[-1]
    response=requests.get(url) 
    with open(filename, 'wb') as f: 
        f.write(response.content) 
    print(f'Downloaded {filename}') 
print('All files downloaded.') 
# move the downloaded files to data/raw/ 
for filename in os.listdir():
    if filename.endswith('.pdf'):
        shutil.move(filename, 'data/raw/')