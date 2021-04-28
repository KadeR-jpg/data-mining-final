import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
uncut_data = pd.read_csv(
    r'C:\Users\KadeC\Desktop\DataMining\DataMine_Code\FinalProject\archive\data.csv')
df = pd.DataFrame(uncut_data)
df.__delitem__('key')
df.__delitem__('id')
training_set = [data for data in df.itertuples() if data.year == 2018 or data.year == 2019]
test_set = [data for data in df.itertuples() if data.year == 2020]
print(len(training_set))
print(len(test_set))
df_train = pd.DataFrame(training_set)
df_test = pd.DataFrame(test_set)
df_test.__delitem__('Index')
df_train.__delitem__('Index')
df_train.to_csv(r'FinalProject/trainData.csv', index=False)
df_test.to_csv(r'FinalProject/testData.csv', index=False)
