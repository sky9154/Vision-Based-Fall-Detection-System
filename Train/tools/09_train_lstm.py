from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense, LSTM
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


LABELS = (0, 1, 2)


def load_dataset (csv_file):
  data = pd.read_csv(csv_file)
  features = data.iloc[:, 2:].to_numpy(dtype=np.float32)
  labels = data.iloc[:, 1].to_numpy(dtype=np.int32)
  found_labels = set(np.unique(labels))

  if found_labels != set(LABELS):
    raise ValueError(f'資料必須包含 0、1、2 三個類別，目前為: {sorted(found_labels)}')

  return features, labels


def create_model (feature_count, hidden_units, learning_rate):
  model = Sequential([
    LSTM(units=hidden_units, input_shape=(feature_count, 1), activation='tanh'),
    Dense(units=3, activation='softmax')
  ])
  model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=categorical_crossentropy,
    metrics=['accuracy']
  )

  return model


def train (csv_file, output, epochs=500, batch_size=128, test_size=0.2, random_state=42, hidden_units=512, learning_rate=0.001):
  features, labels = load_dataset(csv_file)
  x_train, x_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=test_size,
    random_state=random_state,
    stratify=labels
  )

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  y_train_encoded = to_categorical(y_train, num_classes=3)
  y_test_encoded = to_categorical(y_test, num_classes=3)

  output_path = Path(output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  log_path = output_path.with_name(f'{output_path.stem}_training.csv')
  model = create_model(x_train.shape[1], hidden_units, learning_rate)
  callbacks = [
    ModelCheckpoint(str(output_path), monitor='val_loss', save_best_only=True),
    CSVLogger(str(log_path))
  ]

  model.fit(
    x_train,
    y_train_encoded,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_encoded),
    callbacks=callbacks
  )

  best_model = load_model(output_path)
  loss, accuracy = best_model.evaluate(x_test, y_test_encoded, verbose=0)
  predictions = np.argmax(best_model.predict(x_test, verbose=0), axis=1)
  print(f'Test loss: {loss}')
  print(f'Test accuracy: {accuracy}')
  print('Confusion matrix (Safe, Warning, Fall):')
  print(confusion_matrix(y_test, predictions, labels=LABELS))
  print(f'最佳模型: {output_path}')
  print(f'訓練紀錄: {log_path}')


if __name__ == '__main__':
  parser = ArgumentParser(description='訓練安全、警告、跌倒三分類 LSTM 模型')
  parser.add_argument('csv_file', help='合併後的 data.csv 路徑')
  parser.add_argument('--output', default='model/LSTM_03.h5', help='模型輸出路徑')
  parser.add_argument('--epochs', type=int, default=500, help='訓練 Epoch 數')
  parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
  parser.add_argument('--test-size', type=float, default=0.2, help='測試集比例')
  parser.add_argument('--random-state', type=int, default=42, help='隨機種子')
  parser.add_argument('--hidden-units', type=int, default=512, help='LSTM 隱藏單元數')
  parser.add_argument('--learning-rate', type=float, default=0.001, help='學習率')
  args = parser.parse_args()

  train(
    args.csv_file,
    args.output,
    args.epochs,
    args.batch_size,
    args.test_size,
    args.random_state,
    args.hidden_units,
    args.learning_rate
  )
