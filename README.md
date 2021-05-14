


## Multi Tasks(Detect, Track, Dense, Count) in One Frame-Work

A multi task frame-work based on Yolov5+Deepsort, contains:

- [x] **detect task**
- [x] **track task**
- [x] **dense estimate task**
- [x] **boject counting task**

## Demo

**dense estimate demo:**  

![](demo/dense.gif)

**object counting demo:**

![](demo/counter.gif)

**tracking demo(with velocity visulization):**   

![](demo/track.gif)


## Installation

**1.clone this repository**

```
git clone 
cd yolov5_deepsort
```

**2.download yolov5 weights**

```
cd pytorch_yolov5/weights
```

download weights file(yolov5l.pt) from [yolov5 V2.0](https://github.com/ultralytics/yolov5/releases/tag/v2.0) (at the bottom) to this folder.

```
cd ../../
```

**3.download deepsort weights**

```
cd deepsort/deepsort/deep/checkpoint
```

download weights file(ckpt.t7) from [deepsort ckpt](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) (at the bottom) to this folder.

```
cd ../../../../
```

## Usage

```
python main.py --task detect --input {path to images or video or camera} --output {path to result save folder}
                      track
                      dense
                      count
```

more detail parameters can seen in main.py

## References

Thanks for the great work from [[yolov5](https://github.com/ultralytics/yolov5)] and [[deepsort](https://github.com/ZQPei/deep_sort_pytorch)].