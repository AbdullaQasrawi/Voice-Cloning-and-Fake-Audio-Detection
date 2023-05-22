

import os
import random
import shutil
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AudioModel:
    def __init__(self):
        self.files_list = []
        self.dest_path = r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\CommonVoice Dataset\cv-corpus-12.0-delta-2022-12-07\real_audio"
        self.scalar = StandardScaler()
        self.model = None

    def collect_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".mp3"):
                    self.files_list.append(os.path.join(root, file))

    def copy_files_to_destination(self, num_files):
        files_to_copy = random.sample(self.files_list, num_files)
        for file in files_to_copy:
            shutil.move(file, self.dest_path)

    def load_dataframe(self):
        fake_file_list = os.listdir(r'C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\long_version_audio')
        fake_df = pd.DataFrame(fake_file_list)
        fake_df = fake_df.rename(columns={0: 'file'})
        fake_df['real'] = 0
        fake_df.file = [r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\long_version_audio\\" + i for i in
                        fake_df.file]

        real_file_list = os.listdir(
            r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\CommonVoice Dataset\cv-corpus-12.0-delta-2022-12-07\real_audio")
        real_df = pd.DataFrame(real_file_list)
        real_df = real_df.rename(columns={0: 'file'})
        real_df['real'] = 1
        real_df.file = [
            r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\CommonVoice Dataset\cv-corpus-12.0-delta-2022-12-07\real_audio\\" + i
            for i in real_df.file]

        df = pd.concat([real_df, fake_df], axis=0)
        return df

    def extract_features(self, files):
        features = {'mfccs': [], 'chroma': [], 'mel': [], 'contrast': [], 'tonnetz': []}
        for file_name in files:
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            features['mfccs'].append(mfccs)
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            features['chroma'].append(chroma)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            features['mel'].append(mel)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            features['contrast'].append(contrast)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            features['tonnetz'].append(tonnetz)
        return features




    def concatenate_features(self, features):
        features_list = []
        for i in range(len(features['mfccs'])):
            feature_concat = np.concatenate((
                self.scalar.fit_transform((features['mfccs'][i]).reshape(-1, 1)),
                self.scalar.fit_transform((features['chroma'][i]).reshape(-1, 1)),
                self.scalar.fit_transform((features['mel'][i]).reshape(-1, 1)),
                self.scalar.fit_transform((features['contrast'][i]).reshape(-1, 1)),
                self.scalar.fit_transform((features['tonnetz'][i]).reshape(-1, 1))
            ), axis=0)
            features_list.append(feature_concat)
        return np.array(features_list).reshape(len(features['mfccs']), 193)

    def train_model(self, X_train_data, y_train, X_val_data, y_val):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=50, input_shape=(193,), activation='relu'),
            tf.keras.layers.Dense(units=50, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()], optimizer='adam')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        hist = self.model.fit(X_train_data, y_train, validation_data=(X_val_data, y_val), epochs=2000,
                              callbacks=[early_stop])
        return hist

    def save_model_parameters(self, filename):
        self.model.save(filename)


# Usage example:
audio_model = AudioModel()
audio_model.collect_files(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\CommonVoice Dataset\cv-corpus-12.0-delta-2022-12-07\en\clips")
audio_model.copy_files_to_destination(400)
data_frame = audio_model.load_dataframe()
X = data_frame[['file']]
y = data_frame['real']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)
val_features = audio_model.extract_features(X_val.file)
train_features = audio_model.extract_features(X_train.file)
test_features = audio_model.extract_features(X_test.file)
X_val_data = audio_model.concatenate_features(val_features)
X_train_data = audio_model.concatenate_features(train_features)
X_test_data = audio_model.concatenate_features(test_features)
audio_model.train_model(X_train_data, y_train, X_val_data, y_val)
audio_model.save_model_parameters("model_parameters.h5")
