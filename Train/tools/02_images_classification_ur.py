import csv
import os
import shutil
from argparse import ArgumentParser


def images_classification_ur (dataset_path, output='images'):
  output_path = os.path.join(dataset_path, output)

  for label in ('0', '1', '2'):
    os.makedirs(os.path.join(output_path, label), exist_ok=True)

  sources = (
    ('0', 'urfall-cam0-adls.csv'),
    ('1', 'urfall-cam0-falls.csv')
  )

  for source_folder, labels_file in sources:
    labels_path = os.path.join(dataset_path, labels_file)

    with open(labels_path, newline='', encoding='utf-8-sig') as csv_file:
      reader = csv.reader(csv_file)
      next(reader)

      for row in reader:
        folder_number = str(int(row[0].split('-')[1]))
        folder_name = f'{row[0]}-cam0-rgb'
        frame_name = f'{folder_name}-{str(row[1]).zfill(3)}.png'
        label = str(int(row[2]) + 1)
        source = os.path.join(dataset_path, source_folder, folder_number, folder_name, frame_name)

        if os.path.isfile(source):
          shutil.copy2(source, os.path.join(output_path, label, frame_name))


if __name__ == '__main__':
  parser = ArgumentParser(description='依標籤分類 UR Fall Detection Dataset 影像')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--output', default='images', help='輸出目錄名稱，預設為 images')
  args = parser.parse_args()

  images_classification_ur(args.dataset_path, args.output)
