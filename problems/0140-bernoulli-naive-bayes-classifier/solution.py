import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing=smoothing
        self.single_class=None
        self.log_prior={}
        self.log_p1={}
        self.log_p0={}
        self.n_features=None


    def forward(self, X, y):
        # Fit model to binary features X and labels y
        X=np.asarray(X)
        y=np.asarray(y).astype(int)
        m,n=X.shape
        if m!=y.shape[0]:
            raise ValueError("error")

        self.n_features=n
        classes=np.unique(y)
        if classes.size==1:
            self.single_class=int(classes[0])
            return self
        
        self.single_class=None
        alpha=self.smoothing

        for c in (0,1):
            X_c=X[y==c]
            n_c=X_c.shape[0]
            count1=X_c.sum(axis=0)

            p1=(count1+alpha)/(n_c+2*alpha)
            p0=(n_c-count1+alpha)/(n_c+2*alpha)

            self.log_p1[c]=np.log(p1)
            self.log_p0[c]=np.log(p0)

            self.log_prior[c]=np.log(n_c/m)

        return self


    def predict(self, X):
        # Predict class labels for test set X
        X=np.asarray(X,dtype=float)

        n_samples=X.shape[0]
        if self.single_class is not None:
            return np.full(n_samples,self.single_class,dtype=int)
        
        if self.n_features is not None and X.shape[1]!=self.n_features:
            raise ValueError(" not match")
        score0=(self.log_prior[0]+X.dot(self.log_p1[0])+(1.0-X).dot(self.log_p0[0]))
        score1=(self.log_prior[1]+X.dot(self.log_p1[1])+(1.0-X).dot(self.log_p0[1]))

        scores=np.column_stack((score0,score1))

        return np.argmax(scores, axis=1).astype(int)


