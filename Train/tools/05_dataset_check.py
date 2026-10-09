import math
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd


def dataset_check (csv_file):
  data = pd.read_csv(csv_file)
  features = data.drop(columns=['frame', 'labels'])

  print('數據描述性統計量:')
  print(data.describe())

  features.hist(bins=50, figsize=(20, 15))
  plt.show()

  columns = 5
  rows = math.ceil(len(features.columns) / columns)

  features.plot(
    kind='box',
    subplots=True,
    layout=(rows, columns),
    sharex=False,
    sharey=False,
    figsize=(20, rows * 4)
  )
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser(description='檢查數據分佈情形')
  parser.add_argument('csv_file', help='CSV 數據集路徑')

  dataset_check(parser.parse_args().csv_file)
