#add your imports here
from __future__ import division
from __future__ import print_function
# own imports
import ntpath

#custom classes imports
import kerasmodel as km
import segmentor as seg

#original imports
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random



"""
add whatever you think it's essential here
"""
#load keras model
model = km.KerasModel.load()


class SymPred():
	def __init__(self,prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |	   	|
			   |	   	|
			    --------.
			    		 (x2,y2)
		"""
		self.prediction = prediction
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
	def __str__(self):
		return self.prediction + '\t' + '\t'.join([
												str(self.x1),
												str(self.y1),
												str(self.x2),
												str(self.y2)])

class ImgPred():
	def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
		"""
		sym_pred_list is list of SymPred
		latex is the latex representation of the equation
		"""
		self.image_name = image_name
		self.latex = latex
		self.sym_pred_list = sym_pred_list
	def __str__(self):
		res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
		for sym_pred in self.sym_pred_list:
			res += str(sym_pred) + '\n'
		return res

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
	imgs, rects = seg.Segmentor.process(image_path)
	for i in range(0, len(imgs)):
		sym_preds.append(SymPred())

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
