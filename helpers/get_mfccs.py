import os
import math
import json
import librosa
import numpy as np

def get_mfccs(directory_path, fs=22500, duration=30, n_fft=2048, hop_length=512, n_mfcc=13, num_segments=10):
  data = {
    "genre_name": [],
    "genre_num": [],
    "mfcc": []
  }
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
              data["genre_name"].append(genre_current.split('\\')[-1])
              data["genre_num"].append(i-1)
              data["mfcc"].append(mfcc.tolist())
        except:
          continue
      print(f"Collected MFCCs for {genre_current.title()}!")
  with open('./dataRetrain/data.json', "w") as filepath:
    print("========================")
    print("Saving data to disk...")
    json.dump(data, filepath, indent=4)
    print("Saving complete!")
    print("========================")
  return np.array(data["mfcc"]), np.array(data["genre_name"]), np.array(data["genre_num"])

