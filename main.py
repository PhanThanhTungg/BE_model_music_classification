import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "./models/model_cnn3.h5"
model = load_model(model_path)

def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512, segment_duration=3):
    y, sr = librosa.load(file_path, sr=22050)
    
    samples_per_segment = int(segment_duration * sr)
    
    mfccs_result = []
    
    for i in range(0, len(y) - samples_per_segment, samples_per_segment):
        segment = y[i:i + samples_per_segment]
        
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T  
        
        if len(mfcc) == 130:  
            mfccs_result.append(mfcc.tolist())
    
    return np.array(mfccs_result)

def predict_genre(file_path, model):
    mfccs = extract_mfcc(file_path)
    
    if len(mfccs) == 0:
        return "Không thể trích xuất MFCC từ file âm thanh"

    if 'cnn' in model_path:
        mfccs = mfccs[..., np.newaxis]
    
    predictions = model.predict(mfccs)
    avg_prediction = np.mean(predictions, axis=0)
    
    genre_idx = np.argmax(avg_prediction)
    
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_genre = genres[genre_idx]
    
    return predicted_genre, float(avg_prediction[genre_idx])

# Sử dụng hàm dự đoán
file_path = "./1xj558pvss.mp3"
predicted_genre, confidence = predict_genre(file_path, model)
print(f"Thể loại dự đoán: {predicted_genre}")
print(f"Độ tin cậy: {confidence:.2f}")
