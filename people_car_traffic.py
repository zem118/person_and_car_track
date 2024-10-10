import cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont

from Detection import Detection

sys.path.append('../../github/ByteTrack/')
from yolox.tracker.byte_tracker import BYTETracker, STrack

model = torch.hub.load('./yolov5/', 'custom', path='./weights/yolov5s.pt', source='local')
model.conf = 0.1

from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


p_byte_tracker = BYTETracker(BYTETrackerArgs())
c_byte_tracker = BYTETracker(BYTETrackerArgs())


def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./fonts/MSYH.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def track_update(yolo_pd, byte_tracker):
    box_list = yolo_pd.to_numpy()
    detections = []
    for box in box_list:
        l, t = int(box[0]), int(box[1])
        r, b = int(box[2]), int(box[3])

        conf = box[4]

        detections.append([l, t, r, b, conf])

    tracks = byte_tracker.update(
        output_results=np.array(detections, dtype=float),
        img_info=frame.shape,
        img_size=frame.shape
    )

    return tracks


def plot_detection(tracks, pre_detection_dict, cls_name, frame):
    new_detection_dict = {}
    for track in tracks:
        l, t, r, b = track.tlbr.astype(np.int32)
        track_id = track.track_id
        if track_id in pre_detection_dict:
            detection = pre_detection_dict[track_id]
            detection.update((l, t, r, b))
        else:
            detection = Detection(cls_name, (l, t, r, b), track_id)

        new_detection_dict[track_id] = detection

        color = (0, 255, 0) if cls_name == 'person' else (255, 199, 0)
        cv2.rectangle(frame, (l, t), (r, b), color, 1)
        direction = ''
        if detection.direction == 'left':
            direction = '向左行走'
        elif detection.direction == 'right':
            direction = '向右行走'
        cv2.putText(frame, f'{cls_name}-id-{track_id}', (l + 2, t - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        frame = cv2_add_chinese_text(frame, direction, (l + 5, t + 10), (255, 255, 0), 15)

        track_list = detection.get_track_list()
        track_list_np = np.asarray(track_list, np.int32)
        track_list_np = track_list_np.reshape((-1, 1, 2))
        cv2.polylines(frame, [track_list_np], False, (255, 0, 0), 2)
    return new_detection_dict, frame

cap = cv2.VideoCapture('./video_car.mp4')

frame_index = 0
last_person_detection_dict = {}
last_car_detection_dict = {}
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model(frame[:, :, ::-1])
    pd = results.pandas().xyxy[0]
    person_pd = pd[pd['name'] == 'person']
    car_pd = pd[pd['name'] == 'car']

    p_tracks = track_update(person_pd, p_byte_tracker)
    c_tracks = track_update(car_pd, c_byte_tracker)

    last_person_detection_dict, frame = plot_detection(p_tracks, last_person_detection_dict, 'person', frame)
    last_car_detection_dict, frame = plot_detection(c_tracks, last_car_detection_dict, 'car', frame)

    person_num = len(last_person_detection_dict)
    left_num = 0
    right_num = 0
    for track_id, det in last_person_detection_dict.items():
        if det.direction == 'left':
            left_num += 1
        elif det.direction == 'right':
            right_num += 1
    info_frame_h, info_frame_w = 125, 220
    traffic_info_frame = np.zeros((info_frame_h, info_frame_w, 3), np.uint8)
    tif_l, tif_t = 905, 0
    tif_r, tif_b = tif_l+info_frame_w, tif_t+info_frame_h
    frame_part = frame[tif_t:tif_b, tif_l:tif_r]
    mixed_frame = cv2.addWeighted(frame_part, 0.2, traffic_info_frame, 0.8, 0)
    frame[tif_t:tif_b, tif_l:tif_r] = mixed_frame
    frame = cv2_add_chinese_text(frame, f'当前人流量：{person_num}', (tif_l+10, tif_t+10), (255, 128, 0), 30)
    frame = cv2_add_chinese_text(frame, f'向左走的客流：{left_num}', (tif_l+10, tif_t+55), (0, 255, 255), 25)
    frame = cv2_add_chinese_text(frame, f'向右走的客流：{right_num}', (tif_l+10, tif_t+85), (0, 255, 255), 25)

    num = len(last_car_detection_dict)
    left_num = 0
    right_num = 0
    for track_id, det in last_car_detection_dict.items():
        if det.direction == 'left':
            left_num += 1
        elif det.direction == 'right':
            right_num += 1
    traffic_info_frame = np.zeros((info_frame_h, info_frame_w, 3), np.uint8)
    tif_l, tif_t = 905, (info_frame_h + 10)
    tif_r, tif_b = tif_l + info_frame_w, tif_t + info_frame_h
    frame_part = frame[tif_t:tif_b, tif_l:tif_r]
    mixed_frame = cv2.addWeighted(frame_part, 0.2, traffic_info_frame, 0.8, 0)
    frame[tif_t:tif_b, tif_l:tif_r] = mixed_frame
    frame = cv2_add_chinese_text(frame, f'当前车流量：{num}', (tif_l + 10, tif_t + 10), (255, 128, 0), 30)
    frame = cv2_add_chinese_text(frame, f'向左走的车流：{left_num}', (tif_l + 10, tif_t + 55), (0, 255, 255), 25)
    frame = cv2_add_chinese_text(frame, f'向右走的车流：{right_num}', (tif_l + 10, tif_t + 85), (0, 255, 255), 25)

    cv2.imshow('bytetrack', frame)

    frame_index += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
