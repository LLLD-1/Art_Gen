import requests
import os
import json

def stripFileExtension(file):
  return file.split('.')[0]

def getAllJsonInDirectory(directory):
  data = []

  for file in os.scandir(directory):
    path = f'{directory}/{file}'
    with open(path) as f:
        jsonData = json.load(f)
        jsonData['artist_name'] = stripFileExtension(file)
        data.append(json.load(f))

  return data

jsonFiles = getAllJsonInDirectory('../data/links')

for json in jsonFiles:
  artStyle = json['artStyle']

  directoryStyle = f'../data/images/{artStyle}'
  if not os.path.isdir(directoryStyle):
    os.mkdir(directoryStyle)

  directoryStyleArtist = f'../data/images/{artStyle}/{json['artist_name']}'
  if not os.path.isdir(directoryStyleArtist):
    os.mkdir(directoryStyleArtist)
  
  for i, imageLink in enumerate(json['entries']):
    filename = f'{directoryStyleArtist}/{i}.jpg'

    data = requests.get(imageLink).content
    with open(filename, 'w') as f:
      f.write(data)