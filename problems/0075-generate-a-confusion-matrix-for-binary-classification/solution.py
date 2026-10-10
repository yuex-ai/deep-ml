
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	matrix=[]
	tp=fp=tn=fn=0
	for yt,yp in data:
		if yt==1 and yp==1:
			tp+=1
		elif yt==1 and yp==0:
			fn+=1
		elif yt==0 and yp==1:
			fp+=1
		elif yt==0 and yp==0:
			tn+=1
	return [[tp, fn], [fp, tn]]
