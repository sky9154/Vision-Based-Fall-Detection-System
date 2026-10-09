import os
from argparse import ArgumentParser
import pandas as pd


LABELS = ('0', '1', '2')


def data_sampled (dataset_path, source='interpolated', output='sampled', samples=None, random_state=1):
  source_path = os.path.join(dataset_path, source)
  output_path = os.path.join(dataset_path, output)
  datasets = {}

  for label in LABELS:
    csv_path = os.path.join(source_path, f'{label}.csv')

    if not os.path.isfile(csv_path):
      raise FileNotFoundError(f'找不到特徵資料: {csv_path}')

    datasets[label] = pd.read_csv(csv_path)

  available = min(len(data) for data in datasets.values())
  sample_size = available if samples is None else samples

  if sample_size > available:
    raise ValueError(f'每類最多只能抽樣 {available} 筆，目前要求 {sample_size} 筆')

  os.makedirs(output_path, exist_ok=True)

  for label, data in datasets.items():
    sampled = data.sample(n=sample_size, random_state=random_state)
    sampled.to_csv(os.path.join(output_path, f'{label}.csv'), index=False)

  print(f'每個類別已抽樣 {sample_size} 筆資料')


if __name__ == '__main__':
  parser = ArgumentParser(description='平衡安全、警告、跌倒三類特徵資料')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--source', default='interpolated', help='來源目錄名稱，預設為 interpolated')
  parser.add_argument('--output', default='sampled', help='輸出目錄名稱，預設為 sampled')
  parser.add_argument('--samples', type=int, help='每類抽樣數量；省略時使用最少類別的資料量')
  parser.add_argument('--random-state', type=int, default=1, help='隨機種子')
  args = parser.parse_args()

  data_sampled(args.dataset_path, args.source, args.output, args.samples, args.random_state)
