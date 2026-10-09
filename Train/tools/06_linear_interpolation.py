import os
from argparse import ArgumentParser
import pandas as pd


LABELS = ('0', '1', '2')
FEATURE_COLUMNS = [f'{axis}{index}' for axis in ('x', 'y') for index in range(1, 14)]


def interpolate_file (input_path, output_path):
  data = pd.read_csv(input_path)
  missing = [column for column in ['frame', 'labels', *FEATURE_COLUMNS] if column not in data.columns]

  if missing:
    raise ValueError(f'{input_path} 缺少必要欄位: {", ".join(missing)}')

  for column in FEATURE_COLUMNS:
    data[column] = data.groupby('labels')[column].transform(
      lambda values: values.mask(values == 0).interpolate(limit_direction='both').fillna(0)
    )

  data.to_csv(output_path, index=False)


def linear_interpolation (dataset_path, source='features', output='interpolated'):
  source_path = os.path.join(dataset_path, source)
  output_path = os.path.join(dataset_path, output)
  os.makedirs(output_path, exist_ok=True)

  for label in LABELS:
    input_file = os.path.join(source_path, f'{label}.csv')

    if not os.path.isfile(input_file):
      raise FileNotFoundError(f'找不到特徵資料: {input_file}')

    output_file = os.path.join(output_path, f'{label}.csv')
    interpolate_file(input_file, output_file)
    print(f'已輸出: {output_file}')


if __name__ == '__main__':
  parser = ArgumentParser(description='以線性插值補足三分類姿態特徵中的缺失值')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--source', default='features', help='來源目錄名稱，預設為 features')
  parser.add_argument('--output', default='interpolated', help='輸出目錄名稱，預設為 interpolated')
  args = parser.parse_args()

  linear_interpolation(args.dataset_path, args.source, args.output)
