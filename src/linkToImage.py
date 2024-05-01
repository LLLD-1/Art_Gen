import requests
import os
import json

def stripFileExtension(file):
  return file.split('.')[0]

def getAllJsonInDirectory(directory):
  data = []

  for file in os.scandir(directory):
    print(file.path)
    with open(file.path) as f:
        jsonData = json.load(f)
        jsonData['artist_name'] = stripFileExtension(file.name)
        data.append(jsonData)

  return data

jsonFiles = getAllJsonInDirectory('../data/links')

for jsonFile in jsonFiles:
  artStyle = jsonFile['artStyle']

  directoryStyle = f'../data/images/{artStyle}'
  if not os.path.isdir(directoryStyle):
    os.mkdir(directoryStyle)

  directoryStyleArtist = f'../data/images/{artStyle}/{jsonFile["artist_name"]}'
  if not os.path.isdir(directoryStyleArtist):
    os.mkdir(directoryStyleArtist)
  
  for i, imageLink in enumerate(jsonFile['entries']):
    filename = f'{directoryStyleArtist}/{i}.jpg'

    data = requests.get(imageLink).content
    with open(filename, 'wb') as f:
      f.write(data)