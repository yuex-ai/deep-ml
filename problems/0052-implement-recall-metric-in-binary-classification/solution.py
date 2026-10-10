import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    tp=0
    fn=0
    for yt,yp in zip(y_true,y_pred):
        if yt==1 and yp==1:
            tp+=1
        elif yt==1 and yp==0:
            fn+=1
    if tp+fn==0:
        return 0.0
    output=tp/(tp+fn)
    return output
