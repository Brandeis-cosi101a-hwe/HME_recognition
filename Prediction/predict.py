#add your imports here
from __future__ import division
from __future__ import print_function
# own imports
import ntpath

#custom classes imports
import kerasmodel as km
import segmentor2 as seg

#Separate ImgPred and SymPred to files
from SymPred import SymPred
from ImgPred import ImgPred

#original imports
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random

"""
add whatever you think it's essential here
"""
#load keras model and label encoder
model, le = km.KerasModel.load()

def predict(image_path):
	"""
	Add your code here
	"""
	"""
	#Don't forget to store your prediction into ImgPred
	img_prediction = ImgPred(...)
	"""
	sym_preds = []
	print("Predicting "+ ntpath.basename(image_path))
	imgs, rects = seg.process(image_path)
	for i in range(0, len(imgs)):
		sym_pred = SymPred(le.inverse_transform(np.argmax(model.predict(imgs[i].reshape([-1,32,32,1])))),rects[i][0],rects[i][1],rects[i][2],rects[i][3])
		sym_preds.append(sym_pred)
		print(str(sym_pred))
	img_pred = ImgPred(ntpath.basename(image_path).split('.')[0], sym_preds)
	seg.cropped_imgs = []
	seg.cropped_rects = []
	return img_pred

	return img_prediction
if __name__ == '__main__':
	image_folder_path = argv[1]
	isWindows_flag = False
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '\\*png')
	else:
		image_paths = glob(image_folder_path + '/*png')
	results = []
	for image_path in image_paths:
		impred = predict(image_path)
		results.append(impred)

	with open('predictions.txt','w') as fout:
		for res in results:
			fout.write(str(res))
