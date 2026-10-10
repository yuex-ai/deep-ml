import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
    m=len(y_true)
    correct=0
    for yt,yp in zip(y_true,y_pred):
        if yt==yp:
            correct+=1
    acc=correct/m
    return acc