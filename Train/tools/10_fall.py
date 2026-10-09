from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
from keras.models import load_model
from mediapipe import solutions
from PIL import Image, ImageDraw, ImageFont


TRAIN_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1280
HEIGHT = 720


def extract_keypoints (landmarks):
  indices = [0] + list(range(11, 17)) + list(range(23, 29))
  pose_x = [landmarks[i].x * WIDTH for i in indices]
  pose_y = [landmarks[i].y * HEIGHT for i in indices]

  return np.reshape(np.concatenate([pose_x, pose_y]), (1, 26, 1))


def put_text (image, text, position, font_path, font_color=(255, 255, 255), font_size=30):
  image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  draw = ImageDraw.Draw(image)
  font = ImageFont.truetype(str(font_path), font_size, encoding='utf-8')
  draw.text(position, text, font_color, font=font)

  return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def run (model_path, video_path, font_path):
  model = load_model(model_path)

  if model.output_shape[-1] != 3:
    raise ValueError(f'模型必須輸出三個類別，目前為 {model.output_shape[-1]}')

  capture = cv2.VideoCapture(str(video_path))

  if not capture.isOpened():
    raise FileNotFoundError(f'無法開啟影片: {video_path}')

  with solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while True:
      success, frame = capture.read()

      if not success:
        break

      results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

      if not results.pose_landmarks:
        cv2.imshow('MediaPipe Pose', frame)

        if cv2.waitKey(5) == ord('q'):
          break

        continue

      prediction = model.predict(extract_keypoints(results.pose_landmarks.landmark), verbose=0)[0]
      safety, warning, fall = [round(value * 100, 1) for value in prediction]

      solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style()
      )

      frame = put_text(frame, f'安全: {safety}%', (10, 0), font_path)
      frame = put_text(frame, f'警告: {warning}%', (10, 40), font_path)
      frame = put_text(frame, f'跌倒: {fall}%', (10, 80), font_path)

      if warning + fall > 50:
        frame = put_text(frame, '警告', (10, 120), font_path, (255, 0, 0))

      cv2.imshow('MediaPipe Pose', frame)

      if cv2.waitKey(5) == ord('q'):
        break

  capture.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  parser = ArgumentParser(description='使用三分類 LSTM 模型測試跌倒影片')
  parser.add_argument('--model', default=str(TRAIN_ROOT / 'model' / 'LSTM_01.h5'), help='模型路徑')
  parser.add_argument('--video', default=str(TRAIN_ROOT / 'video' / 'fall.mp4'), help='影片路徑')
  parser.add_argument('--font', default=str(TRAIN_ROOT / 'font' / 'NotoSansTC-Medium.otf'), help='中文字型路徑')
  args = parser.parse_args()

  run(args.model, args.video, args.font)
