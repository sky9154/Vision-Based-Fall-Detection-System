import os
from argparse import ArgumentParser
from sys import stdout
from PIL import Image


LABELS = ('0', '1', '2')


def create_variants (image):
  return {
    'original': image.copy(),
    'flip_lr': image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    'flip_tb': image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    'flip_both': image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    'rot45': image.rotate(45, fillcolor='green'),
    'rot135': image.rotate(135, fillcolor='green'),
    'rot225': image.rotate(225, fillcolor='green'),
    'rot315': image.rotate(315, fillcolor='green')
  }


def data_augmentation (dataset_path, source='images', output='augmented'):
  source_path = os.path.join(dataset_path, source)
  output_path = os.path.join(dataset_path, output)

  if not os.path.isdir(source_path):
    raise FileNotFoundError(f'找不到圖片目錄: {source_path}')

  for label in LABELS:
    class_path = os.path.join(source_path, label)

    if not os.path.isdir(class_path):
      raise FileNotFoundError(f'找不到類別目錄: {class_path}')

    class_output = os.path.join(output_path, label)
    os.makedirs(class_output, exist_ok=True)

    images = [name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))]
    total = len(images)

    for index, name in enumerate(images, start=1):
      with Image.open(os.path.join(class_path, name)) as image:
        for prefix, variant in create_variants(image.convert('RGB')).items():
          variant.save(os.path.join(class_output, f'{prefix}_{name}'))

      percentage = round(index / total * 100, 1)
      stdout.write(f'\r【 {label} 】 {index} / {total}: {percentage}%')
      stdout.flush()

    print()


if __name__ == '__main__':
  parser = ArgumentParser(description='依三分類目錄建立幾何增強影像')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--source', default='images', help='來源目錄名稱，預設為 images')
  parser.add_argument('--output', default='augmented', help='輸出目錄名稱，預設為 augmented')
  args = parser.parse_args()

  data_augmentation(args.dataset_path, args.source, args.output)
