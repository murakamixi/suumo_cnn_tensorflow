from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from keras.models import Model
from keras import optimizers
from keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('img_width', 64, 'img_width', short_name='iw')
flags.DEFINE_integer('img_height', 64, 'img_height', short_name='ih')

flags.DEFINE_integer('batch_size', 5, 'batch_size', short_name='b')
flags.DEFINE_integer('epochs', 20, 'epoch', short_name='e')

classes = ['bathroom', 'genkan', 'kitchen', 'mitorizu', 'senmenjo', 'wc',]
nb_classes = len(classes)

img_width, img_height = FLAGS.img_width, FLAGS.img_height

#traning and validation data dir
train_data_dir = './data/train'
validation_data_dir = './data/validation'

batch_size = FLAGS.batch_size
nb_epoch = FLAGS.epochs

result_dir = './result'

def main():
  # トレーンング用、バリデーション用データを生成するジェネレータ作成
  train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=0.2,
    horizontal_flip=True)

  validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

  train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)

  validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True)


  # VGG16のロード。FC層は不要なので include_top=False
  input_tensor = Input(shape=(img_width, img_height, 3))
  vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

  # FC層の作成
  top_model = Sequential()
  top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
  top_model.add(Dense(256, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(nb_classes, activation='softmax'))

  # VGG16とFC層を結合してモデルを作成
  vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

  # 最後のconv層の直前までの層をfreeze
  for layer in vgg_model.layers[:15]:
      layer.trainable = False

  # 多クラス分類を指定
  vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

  # Fine-tuning
  history = vgg_model.fit_generator(
      generator=train_generator,
      epochs=nb_epoch,
      validation_data=validation_generator,
      )

  # 重みを保存
  vgg_model.save_weights(os.path.join(result_dir, 'finetuning.h5'))

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
  plt.show()


if __name__ == '__main__':
  app.run(main)