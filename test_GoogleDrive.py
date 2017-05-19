from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.

drive = GoogleDrive(gauth)

file_id = ''
# # Auto-iterate through all files that matches this query
file_list = drive.ListFile({'q': "title='Airdata.json' and trashed=false"}).GetList()
for file1 in file_list:
    print('title: %s, id: %s' % (file1['title'], file1['id']))
    file_id = file1['id']

test_dict = dict()
test_dict['asd'] = 11
test_dict['qwe'] = 23
test_dict['zxc'] = 34
import json
test_json = json.dumps(test_dict)

if file_id != '':
    file1 = drive.CreateFile({'id': file_id})
    # file1.SetContentString(test_json)  # Set content of the file from given string.
    # file1.Upload()
else:
    file1 = drive.CreateFile({'title': 'Airdata.json',
                              'mimeType': 'application/json'}) # Create GoogleDriveFile instance with title 'Hello.txt'.
    file1.SetContentString(test_json)  # Set content of the file from given string.
    file1.Upload()

file2 = drive.CreateFile({'id': file1['id']})
content = file2.GetContentString()
print(content)
