import os
from argparse import ArgumentParser
from sys import stdout
from PIL import Image, ImageOps


LABELS = ('0', '1', '2')


def image_equalization (dataset_path, source='images', output='preprocessed'):
  source_path = os.path.join(dataset_path, source)
  output_path = os.path.join(dataset_path, output)

  if not os.path.isdir(source_path):
    raise FileNotFoundError(f'找不到圖片目錄: {source_path}')

  for label in LABELS:
    class_path = os.path.join(source_path, label)

    if not os.path.isdir(class_path):
      raise FileNotFoundError(f'找不到類別目錄: {class_path}')

    equalized_path = os.path.join(output_path, 'equalized', label)
    grayscale_path = os.path.join(output_path, 'grayscale', label)
    os.makedirs(equalized_path, exist_ok=True)
    os.makedirs(grayscale_path, exist_ok=True)

    images = [name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))]
    total = len(images)

    for index, name in enumerate(images, start=1):
      with Image.open(os.path.join(class_path, name)) as image:
        rgb_image = image.convert('RGB')
        ImageOps.equalize(rgb_image).save(os.path.join(equalized_path, f'eq_{name}'))
        rgb_image.convert('L').save(os.path.join(grayscale_path, f'gray_{name}'))

      percentage = round(index / total * 100, 1)
      stdout.write(f'\r【 {label} 】 {index} / {total}: {percentage}%')
      stdout.flush()

    print()


if __name__ == '__main__':
  parser = ArgumentParser(description='依三分類目錄建立直方圖均衡化與灰階影像')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--source', default='images', help='來源目錄名稱，預設為 images')
  parser.add_argument('--output', default='preprocessed', help='輸出目錄名稱，預設為 preprocessed')
  args = parser.parse_args()

  image_equalization(args.dataset_path, args.source, args.output)
