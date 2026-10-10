import numpy as np
def precision(y_true, y_pred):
	# Your code here
	m=len(y_pred)
	tp=0
	fp=0
	for yt,yp in zip(y_true,y_pred):
		if yt==1 and yp==1:
			tp+=1
		elif yt==0 and yp==1:
			fp+=1
	if tp+fp==0:
		return 0.0
	output=tp/(tp+fp)
	return output
