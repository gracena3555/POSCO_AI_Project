import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.callbacks import CSVLogger

# Read all npy files
X_data = []
y_data = []
k = 0
for f in os.scandir("./coords/openpose"):
    if f.is_file() and f.name != '.DS_Store':
        x = np.load(f, allow_pickle =True)
        
        # Remove empty coords
        x = [coords for coords in x if 1 in coords.shape]
        x = np.concatenate(x)
        X_data.append(x)
        y_data.extend([k]*x.shape[0])
        k += 1

X_data = np.concatenate(X_data)
y_data = np.array(y_data)

# Sanity check
print("X_data shape: {}".format(X_data.shape))
print("y_data shape: {}".format(y_data.shape))

def scale_transform_normalize(coords):
    """
    Parameters:
    coords (ndarray): array of (x,y,c) coordinates

    Returns:
    ndarray: coords scaled to 1x1 with center at (0,0)
    ndarray: confidence scores of each joint
    """
    coords, scores = coords[:,:,:-1], coords[:,:,-1]
    diff = coords.max(axis=1) - coords.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean = coords.mean(axis=1).reshape(coords.shape[0],1,coords.shape[-1])
    out = (coords - mean) / diff_max
    
    return out, scores

N,D,C = X_data.shape

# Prepare X
X_norm, scores = scale_transform_normalize(X_data)
scores = scores.reshape((N, D, 1))
X_data = np.concatenate([X_norm, scores], axis=2)
X_data /= np.linalg.norm(X_data, axis=2)[:, :, np.newaxis]
X = []

# Prepare y
y = []

# Grab every possible combination of 2 rows
for index in tqdm(combinations(np.arange(N), 2)):
    vec_1 = X_data[index[0]]
    vec_2 = X_data[index[1]]
    X.append(np.concatenate([vec_1, vec_2]).flatten())
    y.append(int(y_data[index[0]] == y_data[index[1]]))

# Downsample majority class
X = np.array(X)
y = np.array(y)
trues = X[y == 1]
falses = X[y == 0]
small_falses = resample(falses, n_samples=trues.shape[0])

balanced_X = np.concatenate([trues, small_falses])
balanced_y = np.concatenate([np.ones((trues.shape[0],)), np.zeros((trues.shape[0]))])

# Split data
X_train, X_test, y_train, y_test = train_test_split(balanced_X, balanced_y, test_size=0.33)

# Define network
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Activation('relu'),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dense(1),
    BatchNormalization(),
    Activation('sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
csv_logger = CSVLogger("model_history_log.csv", append=True)
history = model.fit(x=X_train, y=y_train,
                    batch_size=32, epochs=100,
                    verbose=2, validation_split=0.33,
                    callbacks=[csv_logger])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test
y_pred = model.predict(X_test)

# Check ROC, AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)

# Plot
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()