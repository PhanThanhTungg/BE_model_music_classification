import os
import math
import json
import librosa
import numpy as np

from helpers.connectdtb import get_mongo_client

def get_mfccs(directory_path, fs=22500, duration=30, n_fft=2048, hop_length=512, n_mfcc=13, num_segments=10):
  genres = ['blues', 'classical', 'country', 'disco',
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
  data = {
    "genre_name": [],
    "genre_num": [],
    "mfcc": []
  }
  db = get_mongo_client()
  samples_per_track = fs * duration
  samps_per_segment = int(samples_per_track/num_segments)
  mfccs_per_segment = math.ceil(samps_per_segment/hop_length)
  print("MFCC collection started!")
  print("========================")
  for i, (path_current, folder_names, file_names) in enumerate(os.walk(directory_path)):
    if path_current is not directory_path:
      path_list = path_current.split('/')
      genre_current = path_list[-1]
      for file in file_names:
        file_path = os.path.join(path_current, file).replace(os.sep, '/')
        try:
          audio, fs = librosa.load(file_path, sr=fs)
          for seg in range(num_segments):
            start_sample = seg * samps_per_segment
            end_sample = start_sample + samps_per_segment
            mfcc = librosa.feature.mfcc(
              y=audio[start_sample:end_sample],
              sr=fs,
              n_fft=n_fft,
              hop_length=hop_length,
              n_mfcc=n_mfcc
            )
            mfcc = mfcc.T
            if len(mfcc) == mfccs_per_segment:
              genre_name = genre_current.split('\\')[-1]
              # data["genre_name"].append(genre_name)
              # data["genre_num"].append(genres.index(genre_name))
              # data["mfcc"].append(mfcc.tolist())
              collection = db['mfccs']
              newData = {
                "genre_name": genre_name,
								"genre_num": genres.index(genre_name),
								"mfcc": mfcc.tolist()
							}
              collection.insert_one(newData)
        except:
          continue
      print(f"Collected MFCCs for {genre_current.title()}!")
  return np.array(data["mfcc"]), np.array(data["genre_name"]), np.array(data["genre_num"])

