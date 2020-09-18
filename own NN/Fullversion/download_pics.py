from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)


kds = {}
file_list = drive.ListFile({'q': "'1MZ69Mgh75fSx6KrDHrsSCJo1koG_Qx9x' in parents"}).GetList()
for file1 in file_list:
    if hasNumbers(file1['title']):
        kingdom = int(file1['title'])
        if kingdom not in kds:
            kds[kingdom] = file1['id']

download_or_not = input("Download? (y/n)")
if download_or_not == 'y':
    with tqdm(total=len(kds)) as pbar:
        for kingdom in kds:
            path = 'TestingPictures/' + str(kingdom)
            if not os.path.exists(path):
                os.makedirs(path)
            file_list = drive.ListFile({'q': f"'{kds[kingdom]}' in parents"}).GetList()
            with tqdm(total=len(file_list)) as pbar1:
                for file1 in file_list:
                    ext = file1['title'].split('.')[1]
                    if ext != '.jpg' or ext != '.JPG' or ext != '.png' or ext != '.PNG':
                        continue
                    file6 = drive.CreateFile({'id': file1['id']})
                    my_file = Path(path + '/' + file1['title'])
                    if my_file.is_file():
                        continue
                    try:
                        file6.GetContentFile(path + '/' + file1['title']) # Download file as 'catlove.png'.
                    except:
                        print(file1['title'])
                    pbar1.update(1)
            pbar.update(1)
            pbar1.close()
    pbar.close()
else:
    files = {}
    dirs = os.listdir('TestingPictures/')
    for kingdom in dirs:
        path = 'TestingPictures/' + kingdom + '/'
        for filename in os.listdir(path):
            if filename.endswith(".xlsx"):
                files[filename] = path + filename

    for kingdom in kds:
        file_list = drive.ListFile({'q': f"'{kds[kingdom]}' in parents"}).GetList()
        for file1 in file_list:
            if file1['title'].split('.')[1] == 'xlsx':
                file2 = drive.CreateFile({'id': file1['id']})
                file2.Delete()

    for i in files:
        kingdom = i.split('.')[0].split('_')[0]
        FolderID = kds[int(kingdom)]
        print(kingdom, FolderID)
        file = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": FolderID}]
                            ,'title': i
                            ,'mimeType':'application/vnd.ms-excel'})
        file.SetContentFile(files[i])
        file.Upload({'convert': True})
