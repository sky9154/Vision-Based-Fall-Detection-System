import os
import sys
from argparse import ArgumentParser
from PIL import Image


def images_resize (dataset):
  '''
  將圖片尺寸修改為 (640, 480)
  '''

  for folder in os.listdir(dataset):
    folder_path = os.path.join(dataset, folder)

    if not os.path.isdir(folder_path):
      continue

    path = os.path.join(dataset, folder, 'images')
    imgs = os.path.join(dataset, folder, 'rgb')

    if not os.path.isdir(imgs):
      continue

    os.makedirs(path, exist_ok=True)
    images = [image for image in os.listdir(imgs) if os.path.isfile(os.path.join(imgs, image))]

    for index, image in enumerate(images, start=1):
      source = os.path.join(imgs, image)
      destination = os.path.join(path, f'{str(folder).zfill(4)}_{os.path.basename(image)}')

      with Image.open(source) as img:
        img.resize((640, 480)).save(destination)

      sys.stdout.write(f'\r【 {str(folder).zfill(4)} 】 {round(index / len(images) * 100, 1)}%')
      sys.stdout.flush()

    print()


if __name__ == '__main__':
  parser = ArgumentParser(description='調整圖像大小為 640 × 480')
  parser.add_argument('dataset', help='數據集路徑')

  images_resize(parser.parse_args().dataset)
