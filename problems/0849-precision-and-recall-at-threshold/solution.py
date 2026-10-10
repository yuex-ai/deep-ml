import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code 
    tp=fp=fn=0
    for i in range(len(y_scores)):
        y_scores[i]=np.where(y_scores[i]>=threshold,1,0.0)
    for yt,ys in zip(y_true,y_scores):
        tp+=np.sum((yt==1) and (ys==1))
        fp+=np.sum((yt==0) and (ys==1))
        fn+=np.sum((yt==1) and (ys==0))
    if tp+fp==0:
        precision=0
    else:
        precision=tp/(tp+fp)
    if tp+fn==0:
        recall=0
    else:
        recall=tp/(tp+fn)
    return [precision,recall]
