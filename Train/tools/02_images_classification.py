import csv
import os
import shutil
from argparse import ArgumentParser
from sys import stdout


LABEL_MAP = {
  1: '0',
  2: '1',
  3: '2',
  4: '1',
  5: '2'
}


def images_classification (dataset_path, output='images'):
  output_path = os.path.join(dataset_path, output)

  for label in ('0', '1', '2'):
    os.makedirs(os.path.join(output_path, label), exist_ok=True)

  for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path) or folder == output:
      continue

    images_path = os.path.join(folder_path, 'images')
    labels_path = os.path.join(folder_path, 'labels.csv')

    if not os.path.isdir(images_path) or not os.path.isfile(labels_path):
      continue

    with open(labels_path, newline='', encoding='utf-8-sig') as csv_file:
      rows = list(csv.reader(csv_file))[1:]

    for index, row in enumerate(rows, start=1):
      label = LABEL_MAP.get(int(row[1]))

      if label is None:
        continue

      frame = f'{str(folder).zfill(4)}_rgb_{str(row[0]).zfill(4)}.png'
      source = os.path.join(images_path, frame)

      if os.path.isfile(source):
        shutil.copy2(source, os.path.join(output_path, label, frame))

      stdout.write(f'\r【 {folder} 】 {index} / {len(rows)}')
      stdout.flush()

    print()


if __name__ == '__main__':
  parser = ArgumentParser(description='依標籤分類 Fall Detection Dataset 影像')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--output', default='images', help='輸出目錄名稱，預設為 images')
  args = parser.parse_args()

  images_classification(args.dataset_path, args.output)
