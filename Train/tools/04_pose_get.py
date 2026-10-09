import csv
import os
from argparse import ArgumentParser
from pathlib import Path
from sys import stdout
import cv2
import numpy as np
from mediapipe import solutions


LABELS = ('0', '1', '2')
WIDTH = 1280
HEIGHT = 720
FEATURE_COLUMNS = [f'{axis}{index}' for axis in ('x', 'y') for index in range(1, 14)]


def extract_keypoints (landmarks):
  indices = [0] + list(range(11, 17)) + list(range(23, 29))
  pose_x = [landmarks[i].x * WIDTH for i in indices]
  pose_y = [landmarks[i].y * HEIGHT for i in indices]

  return np.concatenate([pose_x, pose_y])


def normalize_keypoints (keypoints):
  center_x = (keypoints[7] + keypoints[8]) / 2
  center_y = (keypoints[20] + keypoints[21]) / 2
  offset_x = center_x - WIDTH / 2
  offset_y = center_y - HEIGHT / 2

  return np.concatenate([
    keypoints[:13] - offset_x,
    keypoints[13:] - offset_y
  ])


def letterbox (frame):
  height, width = frame.shape[:2]
  scale = min(WIDTH / width, HEIGHT / height)
  resized_width = round(width * scale)
  resized_height = round(height * scale)
  resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
  output = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
  offset_x = (WIDTH - resized_width) // 2
  offset_y = (HEIGHT - resized_height) // 2
  output[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized

  return output


def pose_get (dataset_path, source='images', output='features', normalize=False):
  source_path = Path(dataset_path) / source
  output_path = Path(dataset_path) / output
  output_path.mkdir(parents=True, exist_ok=True)

  with solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for label in LABELS:
      class_path = source_path / label

      if not class_path.is_dir():
        raise FileNotFoundError(f'找不到類別目錄: {class_path}')

      images = sorted(path for path in class_path.iterdir() if path.is_file())
      csv_path = output_path / f'{label}.csv'

      with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame', 'labels', *FEATURE_COLUMNS])

        for index, image_path in enumerate(images, start=1):
          frame = cv2.imread(str(image_path))

          if frame is None:
            print(f'\n略過無法讀取的圖片: {image_path}')
            continue

          results = pose.process(cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB))

          if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark)
            keypoints = normalize_keypoints(keypoints) if normalize else keypoints
          else:
            keypoints = np.zeros(len(FEATURE_COLUMNS))

          writer.writerow([image_path.stem, label, *keypoints])
          percentage = round(index / len(images) * 100, 1)
          stdout.write(f'\r【 {label} 】 {index} / {len(images)}: {percentage}%')
          stdout.flush()

      print()


if __name__ == '__main__':
  parser = ArgumentParser(description='從安全、警告、跌倒三類影像擷取 MediaPipe 姿態特徵')
  parser.add_argument('dataset_path', help='數據集路徑')
  parser.add_argument('--source', default='images', help='來源目錄名稱，預設為 images')
  parser.add_argument('--output', default='features', help='輸出目錄名稱，預設為 features')
  parser.add_argument('--normalize', action='store_true', help='以臀部中心將姿態座標平移至畫面中央')
  args = parser.parse_args()

  pose_get(args.dataset_path, args.source, args.output, args.normalize)
