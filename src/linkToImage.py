import requests
import os
import json

def doesArtistDirExist(artist):
  for styleDir in os.scandir('../data/images'):
    artists = list(os.scandir(styleDir.path))
    artists = [p.name for p in artists]

    if artist in artists:
      return True

  return False
def stripFileExtension(file):
  return '.'.join(file.split('.')[:-1])

def getAllJsonInDirectory(directory):
  data = []

  for file in os.scandir(directory):

    with open(file.path) as f:
        jsonData = json.load(f)
        jsonData['artist_name'] = stripFileExtension(file.name)
        data.append(jsonData)

  return data

jsonFiles = getAllJsonInDirectory('../data/links')

filter_fn = lambda j: not doesArtistDirExist(j['artist_name'])
jsonFiles = filter(filter_fn, jsonFiles)

for jsonFile in jsonFiles:
  print(jsonFile['artist_name'])
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