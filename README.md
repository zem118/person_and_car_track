
## 使用技术
- YOLOv8（You Only Look Once，第八版）是一种以其速度和准确性而闻名的最新对象检测模型。
- ByteTracker是一种先进的跟踪算法，旨在维持对象在帧之间的身份，使其成为执行人数统计等任务的理想选择。



## yoloV8+byteTrack行人与车辆目标检测与追踪
- 支持的模型对于YOLOv8，模型通常根据它们的准确性和速度权衡进行分类。
- 通常支持以下模型：
- YOLOv8n（Nano）：提供高速度和较低的准确性。非常适合处理速度的实时应用
- YOLOv8s（Small）：平衡速度和准确性。适用于许多实际应用。
- YOLOv8m（Medium）：在速度和准确性之间提供良好的权衡。适用于更苛刻的应用
- YOLOv8l（Large）：高准确性，速度较低。最适合准确性为优先考虑的场景。
- （本系统最开始使用yolov5网络实现，后在本地改成yolov8训练，已经将训练好的两个版本权重模型放在weights文件夹中）
## ![image](https://github.com/user-attachments/assets/8f26599f-23e5-40b4-858e-1bdbf0e9e0a4)

## 先决条件
在深入实现之前，请确保您具备以下条件：
- Python 3.10
- Ultralytics

## 设置环境
 ```python
conda create -n person-tracker python==3.10
conda activate person-tracker

```
## 安装必要的库和加载模型
- 解压 datasets.zip 文件，按照步骤训练yolov5模型，不训练也可以，直接用 weights 目录下的模型即可
- 数据集是从 http://humaninevents.org/ 网站下载的某个视频，大家如果需要完整的数据集，可以从该网站下载
- 加载 ByteTrack 目录，需要根据你的 ByteTrack 位置，自行修改路径sys.path.append('../../github/ByteTrack/')
```
pip install ultralytics
git clone https://github.com/ultralytics/yolov5.git
```

## 在yolov5目录下，修改data/coco128.yaml配置文件
```
path: ../datasets/people  # 数据集目录
train: images/train  # 训练集
val: images/train  # 验证集

# Classes
names:
  0: person
  1：car
```
## 修改models/yolov5s.yaml文件中的分类数量
```# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Parameters
nc: 2  # number of classes
```
## 执行训练yolo模型命令
```
python ./train.py --data ./data/coco_people.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --batch-size 30 --epochs 120 --workers 8 --name base_n --project yolo_test

python val.py --data ./data/coco_people.yaml  --weights yolo_test/base_n3/weights/best.pt --batch-size 12

python detect.py --source ./83.jpg --weights yolo_test/base_n3/weights/best.pt --conf-thres 0.3
```
## byteTrack算法
- 各种MOT追踪的API大致类似，先准备目标检测框
```
box_list = yolo_pd.to_numpy()
detections = []
for box in box_list:
    l, t = int(box[0]), int(box[1])
    r, b = int(box[2]), int(box[3])

    conf = box[4]

    detections.append([l, t, r, b, conf])
```
- 这里将识别出的行人检测框，转为numpy结构
```
sys.path.append('../../github/ByteTrack/')
from yolox.tracker.byte_tracker import BYTETracker, STrack

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
    
byte_tracker = BYTETracker(BYTETrackerArgs())

tracks = byte_tracker.update(
        output_results=np.array(detections, dtype=float),
        img_info=frame.shape,
        img_size=frame.shape
    )
```
- 调用ByteTrack的update函数进行匹配，匹配后会给每一个检测框一个唯一的ID。主要思路和核心代码就是这些，基于此在做一些工程编码就可以实现检测和跟踪的效果。
