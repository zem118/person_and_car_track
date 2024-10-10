import cv2
import torch
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont

from Detection import Detection

sys.path.append('../../github/ByteTrack/')
from yolox.tracker.byte_tracker import BYTETracker, STrack

model = torch.hub.load('./yolov5/', 'custom', path='./weights/yolo_people.pt', source='local')
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


byte_tracker = BYTETracker(BYTETrackerArgs())


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

cap = cv2.VideoCapture('./video.mp4')
detection_dict = {}

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model(frame[:, :, ::-1])
    pd = results.pandas().xyxy[0]
    person_pd = pd[pd['class'] == 0]
    box_list = person_pd.to_numpy()
    detections = []
    for box in box_list:
        l, t = int(box[0]), int(box[1])
        r, b = int(box[2]), int(box[3])

        conf = box[4]
        cls_id = box[5]
        if cls_id == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # frame = cv2.rectangle(frame, (l, t), (r, b), color, 2)

        detections.append([l, t, r, b, conf])

    tracks = byte_tracker.update(
        output_results=np.array(detections, dtype=float),
        img_info=frame.shape,
        img_size=frame.shape
    )
    new_detection_dict = {}
    for track in tracks:
        l, t, r, b = track.tlbr.astype(np.int32)
        track_id = track.track_id
        if track_id in detection_dict:
            detection = detection_dict[track_id]
            detection.update((l, t, r, b))
        else:
            detection = Detection('person', (l, t, r, b), track_id)

        new_detection_dict[track_id] = detection

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)
        cv2.putText(frame, f'id-{track_id}', (l+10, t-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        track_list = detection.get_track_list()
        track_list_np = np.asarray(track_list, np.int32)
        track_list_np = track_list_np.reshape((-1, 1, 2))
        cv2.polylines(frame, [track_list_np], False, (255, 0, 0), 2)

    detection_dict = new_detection_dict

    person_num = len(detection_dict)
    left_num = 0
    right_num = 0
    for track_id, det in detection_dict.items():
        if det.direction == 'left':
            left_num += 1
        elif det.direction == 'right':
            right_num += 1
    traffic_info_frame = np.zeros((100, 200, 3), np.uint8)
    tif_l, tif_t = 540, 0
    tif_r, tif_b = tif_l+200, tif_t+100
    frame_part = frame[tif_t:tif_b, tif_l:tif_r]
    mixed_frame = cv2.addWeighted(frame_part, 0.2, traffic_info_frame, 0.8, 0)
    frame[tif_t:tif_b, tif_l:tif_r] = mixed_frame
    frame = cv2_add_chinese_text(frame, f'当前人流量：{person_num}', (tif_l+10, tif_t+10), (255, 128, 0), 25)
    frame = cv2_add_chinese_text(frame, f'向左走的客流：{left_num}', (tif_l+10, tif_t+45), (0, 255, 255), 20)
    frame = cv2_add_chinese_text(frame, f'向右走的客流：{right_num}', (tif_l+10, tif_t+65), (0, 255, 255), 20)

    cv2.imshow('bytetrack', frame)

    frame_index += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()