import os
from pathlib import Path
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gspread_dataframe as gd

full_df = pd.DataFrame(columns=['Power', 'Kingdom', 'Date'])
power, kingdoms, date = [], [], []
dirs = os.listdir('TestingPictures/')
for kingdom in dirs:
	path = 'TestingPictures/' + kingdom
	my_file = Path(path + '/' + kingdom + '_list.xlsx')
	if my_file.is_file():
		kd_df = pd.read_excel(my_file)
		power.append(kd_df['Power'].tolist()[:-1])
		kd_df['Kingdom'] = kingdom
		kingdoms.append(kd_df['Kingdom'].tolist()[:-1])
		date.append(kd_df['Date'].tolist()[:-1])

full_df['Power'] = [j for sub in power for j in sub]
full_df['Kingdom'] = [j for sub in kingdoms for j in sub]
full_df['Date'] = [j for sub in date for j in sub]

# full_df.to_excel('full_df.xlsx')


scope = ['https://spreadsheets.google.com/feeds',
	 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('jsonFileFromGoogle.json', scope)
client = gspread.authorize(creds)

sheet = client.open("full_kingdom_list").sheet1
# print(existing.head())
# existing = gd.get_as_dataframe(sheet)
# existing['Power'] = full_df['Power']
# existing['Kingdom'] = full_df['Kingdom']
# existing['Date'] = full_df['Date']

# existing.drop(existing.index, inplace=True)

# existing = full_df

gd.set_with_dataframe(sheet, full_df, include_index=True)