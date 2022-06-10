from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split

import os
import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('img_width', 64, 'img_width', short_name='iw')
flags.DEFINE_integer('img_height', 64, 'img_height', short_name='ih')

# リサイズする画像のサイズ
img_width, img_height = 64, 64

# img_tagごとに分類するクラスの名前の辞書
img_tag_names = {0:'bathroom', 1:'genkan', 2:'kitchen', 3:'mitorizu', 4:'senmenjo', 5:'wc',}

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())
            ]

def main(argv):
  X = []
  Y = []
  # 全てのfolder名とクラスの数字でフォルダ内に保存されている画像からデータセットを作成する
  for num, folder_name in img_tag_names.items():
      print(num, folder_name)
      # 単一のフォルダーから画像データをデータセット作成のためにone-hotをリストに追加
      for picture in list_pictures('./imgs/{}/'.format(folder_name)):
          img = img_to_array(load_img(picture, target_size=(FLAGS.img_width, FLAGS.img_height)))
          X.append(img)
          Y.append(num)

  # arrayに変換
  X = np.asarray(X)
  Y = np.asarray(Y)

  # 画素値を0から1の範囲に変換
  X = X.astype('float32')
  X = X / 255.0

  # クラスの形式を変換
  Y = np_utils.to_categorical(Y, len(img_tag_names))

  # 学習用データとテストデータ
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

  # npz形式へ書き出し
  np.savez("./data/dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


if __name__ == "__main__":
  app.run(main)