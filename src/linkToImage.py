import requests
import os
import json

def getAllJsonInDirectory(directory):
  data = []

  for file in os.scandir(directory):
    path = f'{directory}/{file}'
    with open(path) as f:
        jsonData = json.load(f)
        jsonData['name'] = file
        data.append(json.load(f))

  return data

jsonFiles = getAllJsonInDirectory('../data/links')

for json in jsonFiles:
  artStyle = json['artStyle']

  directory = f'../data/images/{artStyle}'
  if not os.path.isdir(directory):
    os.mkdir(directory)
  
  for i, imageLink in enumerate(json['entries']):
    filename = f'{directory}/{json['name']}{i}.jpg'

    data = requests.get(imageLink).content
    with open(filename, 'w') as f:
      f.write(data)