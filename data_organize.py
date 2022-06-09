from sys import argv
import pandas as pd
import shutil
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('file_num', 10, 'number of files', short_name='n')

# imgsのcsvから削除するカラムのリスト
drop_house_columns = ['house_id', 'img_id']
# imgsのcsvに含まれているimg_tagで、クラス分類に使用しないimg_tagのリスト
drop_img_tags = ['現地外観写真', 'リビング以外の居室', '駐車場', '前面道路含む現地写真',
        '区画図', 'その他', 'リビング', '収納', 'その他設備', '庭',
        '郵便局', '公園', 'その他内観', 'ショッピングセンター', 'スーパー', '小学校', '中学校', 'コンビニ',
        'ドラッグストア', 'バルコニー', '高校・高専', 'その他現地', '発電・温水設備', '冷暖房・空調設備',
        '住戸からの眺望写真', '防犯設備']
# img_tagごとに分類するクラスの名前の辞書
img_tag_names = {'浴室':'bathroom',
                '玄関':'genkan',
                'キッチン':'kitchen',
                '間取り図':'mitorizu',
                '洗面台・洗面所':'senmenjo',
                'トイレ':'wc',
                }

def main(argv):
  for file_num in range(1, FLAGS.file_num+1):
    imgs_df = pd.read_csv('./csv/suumo/imgs_Yamagata_{}.csv'.format(file_num), index_col=0)
    print(file_num)

    # 不要なカラムを削除
    imgs_df = imgs_df.drop(columns=drop_house_columns)

    # クラス分類に使用しない不要なtagを削除
    for drop_img_tag in drop_img_tags:
      index_name = imgs_df[imgs_df['img_tag'] == drop_img_tag].index
      imgs_df.drop(index_name, inplace=True)

    # 画像のtagごとにディレクトリに分類
    for img_name in imgs_df['img_name']:
      #画像のパス
      img_path = './imgs/suumo/{}.jpg'.format(img_name)
      #どこの画像かのタグを取得
      img_index = imgs_df[imgs_df['img_name']==img_name]['img_tag'].index
      img_tag = imgs_df.loc[img_index[0], 'img_tag']

      #画像のタグによってフォルダ分けをする
      for img_tag_name, folder_name in img_tag_names.items():
        # タグとフォルダ名が一致していればそのフォルダに写真を移動
        if img_tag == img_tag_name:
          try:
            shutil.move(img_path, './imgs/{}/'.format(folder_name))
          except Exception as e:
            move_error = e
            break

if __name__ == '__main__':
  app.run(main)