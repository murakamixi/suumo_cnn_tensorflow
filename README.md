# suumo_cnn_tensorflow
tensorflowでsuumoの画像を分類するネットワークです。

# データセットの作成のために
**自作データセットを使うとき**
　データセットを作成するためには```python data_organize.py --file_num=ファイル数```でデータセットのための画像をそれぞれのラベルのディレクトリに分類することができます。
　データセットをラベルごとのディレクトリに分類することができたら、データセットを作成するために、```create_dataset.py ```を実行します。すると、```data/dataset.npz```が生成されます。
　自作クラスを使って訓練を行う時は最終的に以下のようなディレクトリ構成になっています。

```bash
data
 |--dataset.npz
```

**ファインチューニングを行うとき**
　ファインチューニングを行う際は、```python data_organize.py --file_num=ファイル数 --finetuning=True```とします。すると、ラベルごとに分けられたディレクトリ```./data/train```,```./data/validation```が作成されます。
　ファインチューニングを行う時は最終的に以下のようなディレクトリ構成になっています。

```bash
data
 |--train
 |    |--label1
 |    .
 |    .
 |    .
 |    |--label6
 |
 |--validation
      |--label1
      .
      .
      .
      |--label6
```
　
## データの準備

　スクレイピングしてきた画像をsuumoディレクトリにまとめて、suumoディレクトリごとimgsに追加します。
　画像と同様にcsvディレクトリにcsvをまとめたsuumoディレクトリを追加します。

## 自作クラスで学習を行うためのデータセットの作成

　自作クラスで、学習を行う時は```python data_organize.py --file_num=ファイル数```行います。Colabの時はimgsとcsvを置いたディレクトリまでのパスを記載する必要があります。

### 実行(Local)

　```python data_organize.py --file_num=ファイル数```のようにファイル数を指定して実行すると、画像をimgs以下に作成されたディレクトリに振り分けられ、データセットを作成することができます。

### 実行(Colab)

　Colabで実行する場合は、colabで開き、ディレクトリ名を直接指定することで画像を分類することができます。imgsとcsvを置いたディレクトリまでのパスに```img_path```と```imgs_df```を変更する必要があります。

## ファインチューニングを行うとき
　ファインチューニングを行う際は、```python data_organize.py --file_num=ファイル数 --finetuning=True```とします。すると、ラベルごとに分けられたディレクトリ```./data/train```,```./data/validation```が作成されます。

### 実行(Local)

　```python data_organize.py --file_num=ファイル数```のようにファイル数を指定して実行すると、画像をimgs以下に作成されたディレクトリに振り分けられ、データセットを作成することができます。

### 実行(Colab)

　Colabで実行する場合は、colabで開き、ディレクトリ名を直接指定することで画像を分類することができます。imgsとcsvを置いたディレクトリまでのパスに```img_path```と```imgs_df```を変更する必要があります。

## データセットのnpzを作成
　自作データセットで学習を行う時は、データセットをまとめておいた方が、さまざまなネットワークを試すときに効率的です。そこでデータセットをまとめた```dataset.npz```を作成します。
　作成するためには、```create_dataset.py```を実行します。

　```--img_weight=64 --img_height=64```のように、画像サイズを引数で指定することもできます。

　colabで実行する際は、```create_dataset.ipynb```を使用することで作成することができます。


# 学習
　自作のネットワークでの学習方法と、ファインチューニング、重みを読み込んで確認する方法について紹介します。

## 自作のネットワーク
　自作のネットワークで学習する方法を記載します。
　自作のネットワークで学習するためには、```train,py```を使います。```train,py```は```data/dataset.npz```を読み込んで、自作のCNNで学習を進めます。
　引数は以下の通りです。

|引数=初期値|ショート|タイプ|概要|
|:--|:--:|:--:|:--|
|img_width=64|iw|int|画像の横幅|
|img_height=64|ih|int|画像の高さ|
|batch_size=5|b|int|バッチサイズ|
|epochs=20|e|int|エポック数|

## ファインチューニング
　ファインチューニングを行って学習を進めるときは、```fine_tuning.py```を使います。```fine_tuning.py```では、```./data/train```と```./data/validation```ディレクトリの分類画像データを使用します。```fine_tuning.py```ではファインチューニングの一例として、VGG16を使ったコード例を紹介しています。
　学習結果の保存は、```./result/finetuning.h5```に保存され、```load_weight.py```を使って重みをロードして予測することができます。
　学習結果の保存形式は、.h5拡張子のファイルです。.h5拡張子のファイルは、階層型に情報が保存されていて、.pickleなどのバイナリーデータではできない一部の書き換えや、読み込みができるという利点があります。また、.h5拡張子のファイルは言語に依存せずに利用できると言う利点もあります。

　引数は以下の通りです。

|引数=初期値|ショート|タイプ|概要|
|:--|:--:|:--:|:--|
|img_width=64|iw|int|画像の横幅|
|img_height=64|ih|int|画像の高さ|
|batch_size=5|b|int|バッチサイズ|
|epochs=20|e|int|エポック数|

# 予測
　今回は、ファインチューニングの時のみ紹介します。自作ネットワークの時も大差ないはずです。
　```load_weight.py```を使って```./result/finetuning.h5```の重みファイルををロードして、ファインチューニングしたネットワークを使って予測することができます。
　予測するための画像は、```--file_name```で指定できます。
　出力は、画像と以下の通りです。```pred label```は予測されたラベルを表しています。

```bash
input file name : ./data/train/bathroom/11_7.jpg
pred label : bathroom
```

　引数は以下の通りです。


|引数=初期値|ショート|タイプ|概要|
|:--|:--:|:--:|:--|
|img_width=64|iw|int|画像の横幅|
|img_height=64|ih|int|画像の高さ|
|file_name='./data/train/bathroom/11_7.jpg'|f|str|画像サイズ|
|epochs=20|e|int|エポック数|