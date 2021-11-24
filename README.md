# Smart_ATM

## Description

We propose a **Smart ATM model** that can prevent voice phishing face-to-face defraudation damage in ATMs.

- First, we calculate the risk through **personal withdrawal details, loans received from banks, insurance loans, and card loan information**.

- Second, based on the ***yolov5*** algorithm, masks, hands, and faces are detected to determine whether to **wear a mask or not**, and whether to make a **call** is determined.

- Third, based on ***Efficienet-b4***, **facial expressions are recognized** to determine whether they are embarrassed or anxious.

- Finally, **the risk** calculated for each is added to determine whether there is voice phishing.

More detailed description can be found through <a href='https://github.com/winston1214/project/blob/master/AI_HUB_IDEA_CONTEST2021/(Public)%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EA%B3%B5%EB%AA%A8%EC%A0%84%20%EC%95%84%EC%9D%B4%EB%94%94%EC%96%B4%20%EA%B8%B0%ED%9A%8D%EC%84%9C_%ED%94%BC%EC%8B%B1%20%ED%97%8C%ED%84%B0.pdf'>this link</a>

## Member(Pishing Hunter)

<table>
  <tr>
      <td align="center"><a href="https://github.com/winston1214"><img src="https://avatars.githubusercontent.com/u/47775179?v=4" width="100" height="100"><br /><sub><b>김영민</b></sub></td>
      <td align="center"><a href="https://github.com/yeclairer"><img src="https://avatars.githubusercontent.com/u/72565037?v=4" width="100" height="100"><br /><sub><b>곽윤경</b></sub></td>
      <td align="center"><a href="https://github.com/YEO1214"><img src="https://avatars.githubusercontent.com/u/68039225?v=4" width="100" height="100"><br /><sub><b>여지민</b></sub></td>
      <td align="center"><a href="https://github.com/YangDongJae"><img src="https://avatars.githubusercontent.com/u/59679706?v=4" width="100" height="100"><br /><sub><b>양동재</b></sub></td> 
     </tr>
</table>

## OS
- Ubuntu 18.04.5LTS, Tesla V100-SXM2 32GB

## Dataset

- Facial Recognition : <a href='https://aihub.or.kr/aidata/27716'>AI hub 한국인 표정인식 데이터셋</a>
- Mask Detection : <a href='https://www.kaggle.com/andrewmvd/face-mask-detection'>Kaggle Mask Dataset</a>
- Hand Detection : <a href='https://www3.cs.stonybrook.edu/~cvl/projects/hand_det_attention/'>COCO-Hand Dataset</a>
- My data(Sample Data) : <a href='https://github.com/winston1214/Smart_ATM/blob/master/data/data.xlsx'>Individual Information(we make)</a>


## How to do?(Demo)

```
$ git clone https://github.com/winston1214/Smart_ATM.git && cd Smart_ATM
```
```
$ pip install -r requirements.txt
```
```
$ python detect.py --source ${VIDEO_PATH} --weights weights/detection_best.pt --facial-weights-file weights/facial_best.pt --id ${user number}
```

## Training

**Object Detection Training**

This is the same as **yolov5** method.

- Original yolov5 github : <a href='https://github.com/ultralytics/yolov5'>yolov5</a>

- Train description(Korean ver) : <a href='https://bigdata-analyst.tistory.com/195?category=883085'>My Blog</a>

- Train description(Eng ver) : <a href='https://github.com/winston1214/AICT/tree/master/yolov5#readme'>My github readme</a>

**Facial Recognition**

1. Setting dataset(Image) → Crop the face part of a person's image.
2. Seeting dataset(Label) → Make a csv file with two columns(image name, label).  **※ I set [normal : 0, danger : 1]**
3. change directory ```$ cd facial_recognition```
4. run
 ```
 $ python facial_train.py --root ${image root} --csv ${label csv} --batch {Batch-size} --epochs {Number of epoch} --lr {learning rate}
```

