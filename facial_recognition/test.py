from facial_model import Facial_model
import torch
import cv2
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Facial_model().to(device)
model.load_state_dict(torch.load('weights/best.pt'))
img = cv2.imread('_M_emb_20201202171820-001-013.jpg')
image = cv2.resize(img, dsize=(640,640),interpolation=cv2.INTER_LINEAR)
image_swap = np.swapaxes(image, 0,2)
image_swap = np.expand_dims(image_swap, axis=0)
tensor = torch.from_numpy(image_swap).type(torch.FloatTensor).to(device)
model.eval()
print(model(tensor))