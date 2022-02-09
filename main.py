import pandas as pd
import librosa
import numpy as np
import os
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Data/features_30_sec.csv')
pd.set_option('display.max_colwidth', None)
print(data.head(10))

print(data['label'].value_counts())

# data preprocessing
def extract(file):
    try:
        y, sr = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccsScaled = np.mean(mfccs.T, axis=0)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return mfccsScaled, chroma_stft, spectral_bandwidth
    except:
        print("Corrupted file: " + file)
        return




extracted = []

# iterate through each folder and file and get sample rate,label, and mfccs + other features
if not path.exists('extracted.pkl'):
    for index, row in tqdm(data.iterrows()):
        genre = str(row['filename'])[:str(row['filename']).index('.')]
        filen = os.path.join(os.path.abspath('Data/genres_original/'), genre, str(row['filename']))
        label = row['label']
        try:
            mfccs_scaled, chroma_stft, spectral_bandwidth = extract(filen)
            extracted.append([mfccs_scaled, chroma_stft, spectral_bandwidth, label])
        except TypeError:
            continue
    df = pd.DataFrame(extracted, columns=['MFCCS_Scaled','Chroma_STFT', 'Spectral_Bandwidth', 'Genre'])
    df.to_pickle('extracted.pkl')
else:
    df = pd.read_pickle('extracted.pkl')



le = LabelEncoder()

genres = {
    0:'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock'
}
y = np.array(le.fit_transform(df['Genre']))

print(df["Spectral_Bandwidth"])
X = df["MFCCS_Scaled"].to_list()
X = np.array(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

model = LogisticRegression()
model.fit(X_train, y_train)


print(model.score(X_test, y_test))
mfccs_scaled, chroma_stft, spectral_bandwidth = extract('jazz.wav')
x = np.array(mfccs_scaled)
print(x)
print(genres[model.predict(np.array(x).reshape(1,-1))[0]])