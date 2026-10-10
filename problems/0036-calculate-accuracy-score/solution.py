import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
    m=len(y_true)
    correct=0
    for i in range(m):
        if y_true[i]==y_pred[i]:
            correct+=1
    acc=correct/m
    return acc