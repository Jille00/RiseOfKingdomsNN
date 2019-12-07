import cv2
import pytesseract
from glob import glob
import numpy as np
import sys

img_mask = 'Pictures/*.jpg'
img_names = glob(img_mask)
text = []
conf = "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789"
for fn in img_names:
	img = cv2.imread(fn)
	img = img[450:1400, 1900:2150]
	img1 = img[50:110, 20:210]
	text.append(pytesseract.image_to_string(img1, lang='eng', config=conf))
	img2 = img[200:270, 20:210]
	text.append(pytesseract.image_to_string(img2, lang='eng', config=conf))
	img3 = img[380:420, 20:210]
	text.append(pytesseract.image_to_string(img3, lang='eng', config=conf))
	img4 = img[530:600, 20:210]
	text.append(pytesseract.image_to_string(img4, lang='eng', config=conf))
	img5 = img[690:760, 20:210]
	text.append(pytesseract.image_to_string(img5, lang='eng', config=conf))
	img6 = img[860:920, 20:210]
	text.append(pytesseract.image_to_string(img6, lang='eng', config=conf))

new_text = []
for i in text:
	new_i = i.replace(',', '')
	try:
		new_text.append(int(new_i))
	except:
		new_text.append(new_i)
x = 1
y = -1
wrong = 0
for i,j in enumerate(new_text):
	if isinstance(j, int):
		pass
	else:
		wrong += 1
		forward = new_text[i+x]
		backwards = new_text[i+y]
		while not isinstance(forward, int):
			x += 1
			forward = new_text[i+x]
		while not isinstance(backwards, int):
			y -= 1
			backwards = new_text[i+y]
		new_text[i] = int(np.mean([forward,backwards]))
		x = 1
		y = -1

total = 0
for i in new_text:
	total += i
print(total)
print(wrong)