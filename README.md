# Smart_ATM

## Description

We propose a Smart ATM model that can prevent voice phishing face-to-face defraudation damage in ATMs.

## Dataset

- Facial Recognition : <a href='https://aihub.or.kr/aidata/27716'>AI hub 한국인 표정인식 데이터셋</a>
- Mask Detection : <a href='https://www.kaggle.com/andrewmvd/face-mask-detection'>Kaggle Mask Dataset</a>
- Hand Detection : <a href='https://www3.cs.stonybrook.edu/~cvl/projects/hand_det_attention/'>COCO-Hand Dataset</a>
- My data(Sample Data) : <a href='https://github.com/winston1214/Smart_ATM/blob/master/data/data.xlsx'>Individual Information(we make)</a>


## How to do?(Demo)

```
$ git clone https://github.com/winston1214/Smart_ATM.git
```
```
$ pip install -r requirements.txt
```
```
$ python detect.py --source ${VIDEO_PATH} --weights best.pt
```
