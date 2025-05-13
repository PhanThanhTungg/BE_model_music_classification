import numpy as np
import librosa
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import zipfile
import shutil
from helpers.get_mfccs import get_mfccs

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2

from helpers.connectdtb import get_mongo_client
db = get_mongo_client()
app = Flask(__name__)
CORS(app)

model_path = "./models/model_cnn3.h5"
model = load_model(model_path)

def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512, segment_duration=3):
    y, sr = librosa.load(file_path, sr=22050)

    samples_per_segment = int(segment_duration * sr)
    mfccs_result = []

    for i in range(0, len(y) - samples_per_segment, samples_per_segment):
        segment = y[i:i + samples_per_segment]
        mfcc = librosa.feature.mfcc(
            y=segment, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == 130:
            mfccs_result.append(mfcc.tolist())

    return np.array(mfccs_result)


def predict_genre(file_path, model):
    mfccs = extract_mfcc(file_path)

    if len(mfccs) == 0:
        return "Không thể trích xuất MFCC từ file âm thanh", 0.0

    if 'cnn' in model_path:
        mfccs = mfccs[..., np.newaxis]

    predictions = model.predict(mfccs)
    avg_prediction = np.mean(predictions, axis=0)

    genre_idx = np.argmax(avg_prediction)

    genres = ['blues', 'classical', 'country', 'disco',
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    predicted_genre = genres[genre_idx]

    return predicted_genre, float(avg_prediction[genre_idx])


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    if file:
        temp_dir = tempfile.mkdtemp()
        try:
            audio_path = os.path.join(temp_dir, file.filename)
            file.save(audio_path)
            
            zip_path = os.path.join(temp_dir, 'audio.zip')
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                zip_file.write(audio_path, os.path.basename(audio_path))
            
            extract_dir = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            extracted_file_path = os.path.join(extract_dir, os.path.basename(audio_path))
            
            genre, confidence = predict_genre(extracted_file_path, model)
            return jsonify({
                'genre': genre,
                'confidence': float(confidence)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/retrain', methods=['POST'])
def retrain():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, 'uploaded.zip')
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        genres_dir = None
        for root, dirs, files in os.walk(temp_dir):
            for d in dirs:
                if d == 'genres_original':
                    genres_dir = os.path.join(root, d)
                    get_mfccs(genres_dir)
                    break
            if genres_dir:
                break

        if not genres_dir:
            return jsonify({'error': 'Không tìm thấy thư mục genres_original trong file zip'}), 400

        genre_num = []
        mfcc = []

        print("Loading MFCC data from MongoDB...")
        mfcc_data = db['mfccs']
        print("Loading data from MongoDB...")
        for doc in mfcc_data.find():
            genre_num.append(doc['genre_num'])
            mfcc.append(doc['mfcc'])
        print("Data loaded successfully!")

        X = np.array(mfcc)
        y = np.array(genre_num)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

        X_train_cnn = X_train[..., np.newaxis]
        X_val_cnn = X_val[..., np.newaxis]
        X_test_cnn = X_test[..., np.newaxis]
        input_shape = X_train_cnn.shape[1:4]

        model_cnn3 = Sequential()

        # Create a convolution block
        model_cnn3.add(Conv2D(32, 3, activation='relu', input_shape=input_shape)) # first hidden conv layer
        model_cnn3.add(BatchNormalization())
        model_cnn3.add(MaxPooling2D(3, strides=(2,2), padding='same')) # MaxPool the results
        model_cnn3.add(Dropout(0.2))

        # Add another conv block
        model_cnn3.add(Conv2D(64, 3, activation='relu'))
        model_cnn3.add(BatchNormalization())
        model_cnn3.add(MaxPooling2D(3, strides=(2,2), padding='same'))
        model_cnn3.add(Dropout(0.1))

        # Add another conv block
        model_cnn3.add(Conv2D(64, 2, activation='relu'))
        model_cnn3.add(BatchNormalization())
        model_cnn3.add(MaxPooling2D(2, strides=(2,2), padding='same'))
        model_cnn3.add(Dropout(0.1))

        # Flatten output to send through dense layers
        model_cnn3.add(Flatten())
        model_cnn3.add(Dense(128, activation='relu'))
        model_cnn3.add(Dropout(0.5))

        # output to 10 classes for predictions
        model_cnn3.add(Dense(10, activation='softmax'))

        model_cnn3.compile(
            optimizer=Adam(learning_rate=0.0001), # can also use 'adam'
            loss='sparse_categorical_crossentropy', # loss for multi-class classification
            metrics=['acc']
        )

        datagen = ImageDataGenerator(vertical_flip=True)
        es_cnn3 = EarlyStopping(monitor='val_loss', patience=20, min_delta=0) 

        print("Training CNN model...")
        model_cnn3.fit(
            datagen.flow(X_train_cnn, y_train),
            validation_data=(X_val_cnn, y_val),
            batch_size=64,
            epochs=400,
            verbose=1,
            callbacks=[es_cnn3]
        )
        print("Training finished!")
        model_cnn3.save('./models/model_cnn3.h5')

        loss_cnn3, acc_cnn3 = model.evaluate(X_test_cnn, y_test)

        return jsonify({
            'loss': float(loss_cnn3),
            'accuracy': float(acc_cnn3)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(debug=True)
