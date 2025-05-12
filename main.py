import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import zipfile
import shutil
from helpers.get_mfccs import get_mfccs

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
        temp = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp.name)

        try:
            genre, confidence = predict_genre(temp.name, model)

            return jsonify({
                'thể loại': genre,
                'độ tin cậy': float(confidence)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Xóa file tạm
            temp.close()
            os.unlink(temp.name)
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
        # Chỉ lưu folder có tên là genres_original vào ./uploaded_data
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

        project_upload_dir = os.path.join(os.path.dirname(__file__), 'uploaded_data')
        os.makedirs(project_upload_dir, exist_ok=True)
        saved_dir = os.path.join(project_upload_dir, os.path.basename(genres_dir))
        if os.path.exists(saved_dir):
            shutil.rmtree(saved_dir)
        shutil.copytree(genres_dir, saved_dir)


        audio_files = []
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith('.mp3') or f.lower().endswith('.wav'):
                    audio_files.append(os.path.join(root, f))

        if not audio_files:
            return jsonify({'error': 'Không tìm thấy file mp3 hoặc wav trong file zip'}), 400

        return jsonify({'số lượng file audio': len(audio_files), 'danh sách': audio_files})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    app.run(debug=True)
# set FLASK_APP=main.py  
# set FLASK_ENV=development
# flask --app main.py run
# flask run