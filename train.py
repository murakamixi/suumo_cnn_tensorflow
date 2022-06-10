from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('img_width', 64, 'img_width', short_name='iw')
flags.DEFINE_integer('img_height', 64, 'img_height', short_name='ih')

flags.DEFINE_integer('batch_size', 5, 'batch_size', short_name='b')
flags.DEFINE_integer('epochs', 20, 'epoch', short_name='e')

IMAGE_SHAPE = (FLAGS.img_width, FLAGS.img_height, 3)
NUM_CLASSES = 6             # 出力は0~5の6クラス

def cnn(input_shape, num_classes):
    model = Sequential()
    # 隠れ層:16、入力層:データサイズ、活性化関数:Relu
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    # 出力層:分類するクラス数、活性化関数:Softmax
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model

def main(argv):
  # データセットの準備
  datasets = np.load("./data/dataset.npz")

  X_train = datasets['X_train']
  X_test = datasets['X_test']
  y_train = datasets['y_train']
  y_test = datasets['y_test']

  model = cnn(IMAGE_SHAPE, NUM_CLASSES)

  # モデルをコンパイル
  model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

  history = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs, validation_split=0.2,
          # callbacks=[TensorBoard(log_dir=log_dir),
          #           EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
          #           ],
          validation_data = (X_test, y_test),
          verbose=1,
          )
  # historyの中には学習の履歴があるので、確認する。
  # 以下では、学習データとテストデータのaccuracyをプロットしている。

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
  plt.show()

  # テストデータに適用
  predict_classes = model.predict_classes(X_test)

  # マージ。yのデータは元に戻す
  mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(y_test, axis=1)})

  # confusion matrix
  pd.crosstab(mg_df['class'], mg_df['predict'])

if __name__ == '__main__':
  app.run(main)