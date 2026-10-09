from fastapi import WebSocket
from mediapipe import solutions
from keras.models import load_model
from dotenv import load_dotenv
import numpy  as np
import cv2
import os
from functions import device as Device


load_dotenv()

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_pose = solutions.pose

model = None
pose = None
state = False

THRESHOLD = int(os.getenv('THRESHOLD'))

async def extract_keypoints (results) -> tuple[list, float]:
  '''
  提取關鍵點
  '''

  landmarks = results.landmark

  indices = [0] + list(range(11, 17)) + list(range(23, 29))

  pose_x = [landmarks[i].x * 1280 for i in indices]
  pose_y = [landmarks[i].y * 720 for i in indices]
  visibility = [landmarks[i].visibility for i in indices]

  average_visibility = sum(visibility) / len(visibility)

  return np.concatenate([pose_x, pose_y]), average_visibility


async def process_image (frame):
  '''
  調整影像大小，並進行姿勢估計
  '''

  height, width = frame.shape[:2]
  aspect_ratio = width / height

  new_width = int(height * 16 / 9)
  new_aspect_ratio = 16 / 9

  if abs(aspect_ratio - new_aspect_ratio) > 0.001:
    pad = int((new_width - width) / 2)

    image = np.zeros((height, new_width, 3), np.uint8)
    image[:, pad : pad + width] = frame
  else:
    image = frame

  resized = cv2.resize(image, (1280, 720), interpolation = cv2.INTER_AREA)
  resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

  return image


async def stream (websocket: WebSocket, camera_id: str, draw: bool):
  '''
  傳輸影像串流
  '''

  global model, pose, state

  cap = None
  state=False

  number = 0
  safety = 0
  warning = 0
  fall = 0

  if model is None:
    model = load_model('model/LSTM_01.h5')

  if pose is None:
    pose = mp_pose.Pose(
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7,
      model_complexity=0
    )

  device = await Device.get(camera_id)
  device = device[0]

  if device['type'] == 'demo':
    cap = cv2.VideoCapture(f'video/{device["camera"]}')
  elif device['type'] == 'camera':
    cap = cv2.VideoCapture(f'rtsp://{device["camera"]}/mjpeg/1')
  else:
    cap = cv2.VideoCapture(0)

  while True:
    try:
      success, frame = cap.read()

      if not success:
        break

      frame = await process_image(frame)

      results = pose.process(frame)

      output = frame if draw else cv2.imread('assets/images/paper.png')

      mp_drawing.draw_landmarks(
        output,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
      )

      pose_landmarks = results.pose_landmarks
      average_visibility = 0

      if pose_landmarks:
        keypoints, average_visibility = await extract_keypoints(pose_landmarks)
        keypoints = np.reshape(keypoints, (1, 26, 1))

        average_visibility = round(average_visibility * 100, 1)

        if average_visibility >= THRESHOLD:
          prediction = model.predict(keypoints, verbose=0)

          safety = round(prediction[0][0] * 100, 1)
          warning = round(prediction[0][1] * 100, 1)
          fall = round(prediction[0][2] * 100, 1)

      if (safety < 90) != state and not state:
        number = 0
        state = True

        print(f'INFO:     可信值: {average_visibility} %')

        cv2.imwrite('assets/images/fall.png', frame)

      if safety > 90:
        number += 1

      if number > 10:
        number = 0
        state = False


      success, buffer = cv2.imencode('.jpg', output)
      frame = buffer.tobytes()

      await websocket.send_bytes(frame)
      await websocket.send_json({
        'predict': {
          'safety': safety,
          'warning': warning,
          'fall': fall
        },
        'state': state
      })
    except Exception as e:
      print(f'ERROR:    {e}')

      break
