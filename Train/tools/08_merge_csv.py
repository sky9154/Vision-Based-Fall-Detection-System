from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ['frame', 'labels', *[f'{axis}{index}' for axis in ('x', 'y') for index in range(1, 14)]]


def merge_csv (csv_files, output, random_state=42):
  datasets = []

  for csv_file in csv_files:
    data = pd.read_csv(csv_file)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing:
      raise ValueError(f'{csv_file} 缺少必要欄位: {", ".join(missing)}')

    datasets.append(data[REQUIRED_COLUMNS])

  merged = pd.concat(datasets, ignore_index=True)
  labels = set(merged['labels'].astype(int).unique())

  if labels != {0, 1, 2}:
    raise ValueError(f'資料必須包含 0、1、2 三個類別，目前為: {sorted(labels)}')

  merged = merged.sample(frac=1, random_state=random_state).reset_index(drop=True)
  output_path = Path(output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  merged.to_csv(output_path, index=False)
  print(f'已合併 {len(merged)} 筆資料至 {output_path}')


if __name__ == '__main__':
  parser = ArgumentParser(description='合併三分類姿態特徵 CSV')
  parser.add_argument('csv_files', nargs='+', help='需要合併的 CSV 路徑')
  parser.add_argument('--output', default='data.csv', help='輸出路徑，預設為 data.csv')
  parser.add_argument('--random-state', type=int, default=42, help='隨機種子')
  args = parser.parse_args()

  merge_csv(args.csv_files, args.output, args.random_state)
