import pandas as pd
import librosa
import numpy as np
import os
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



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

data = pd.read_csv('../musicML/features_30_sec.csv')
pd.set_option('display.max_colwidth', None)


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
if not path.exists('../musicML/extracted.pkl'):
    for index, row in tqdm(data.iterrows()):
        genre = str(row['filename'])[:str(row['filename']).index('.')]
        filen = os.path.join(os.path.abspath('../musicML/Data/genres_original/'), genre, str(row['filename']))
        label = row['label']
        try:
            mfccs_scaled, chroma_stft, spectral_bandwidth = extract(filen)
            extracted.append([mfccs_scaled, chroma_stft, spectral_bandwidth, label])
        except TypeError:
            continue
    df = pd.DataFrame(extracted, columns=['MFCCS_Scaled','Chroma_STFT', 'Spectral_Bandwidth', 'Genre'])
    df.to_pickle('extracted.pkl')
else:
    df = pd.read_pickle('../musicML/extracted.pkl')








le = LabelEncoder()
y_ = df['Genre']
y = np.array(le.fit_transform(df['Genre']))

X = df["MFCCS_Scaled"].to_list()
X = np.array(X)




# training and testing the ORIGINAL data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', max_iter=999)
model_scores = {}
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['Logistic_Regression'] = temp_scores
print("Logistic Regression: ", model.score(X_test, y_test))

from sklearn import svm
model = svm.SVC(max_iter=999)
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['SVM'] = temp_scores
print("SVM: ", model.score(X_test, y_test))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


TPR = TP/(TP+FN)

TNR = TN/(TN+FP)

PPV = TP/(TP+FP)

NPV = TN/(TN+FN)

FPR = FP/(FP+TN)

FNR = FN/(TP+FN)

FDR = FP/(TP+FP)

ACC = (TP+TN)/(TP+FP+FN+TN)
print(FNR, FPR)
print(classification_report(y_test,model.predict(X_test), target_names=genres.values()))



from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['MLP_Classifier'] = temp_scores
print("MLP: ", model.score(X_test, y_test))

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['KNN'] = temp_scores
print("KNN: ", model.score(X_test, y_test))

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=999, use_label_encoder=False)
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['XGB_Classifier'] = temp_scores
print("XGB: ", model.score(X_test,y_test))

print(classification_report(y_test,model.predict(X_test), target_names=genres.values()))
cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


TPR = TP/(TP+FN)

TNR = TN/(TN+FP)

PPV = TP/(TP+FP)

NPV = TN/(TN+FN)

FPR = FP/(FP+TN)

FNR = FN/(TP+FN)

FDR = FP/(TP+FP)

ACC = (TP+TN)/(TP+FP+FN+TN)
print(FNR, FPR)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


model = MultinomialNB()
temp_scores = []
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    temp_scores.append(model.score(X_test, y_test))

model_scores['Naive_Bayes'] = temp_scores
print(model.score(X_test, y_test))
import pprint
pprint.pprint(model_scores)

