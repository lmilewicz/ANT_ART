import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

class BinaryMemebershipMatrix:
    def __init__(self, vectors, labels):
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)
        flat_labels = labels.T.reshape([-1])


        bmm = np.eye(np.max(flat_labels) + 1)[flat_labels]
        bmm = np.hstack(np.vsplit(bmm, np.max(labels) + 1)).T
        self.data = (bmm * np.linalg.norm(vectors, axis=1)).T

    def __array__(self):
        return self.data

    def __str__(self):
        return str(np.asarray(self))

class ART:
    def __init__(self):
        self.rsnns = importr('RSNNS')
        self.predict = robjects.r('predict')

    def cluster(self, bmm: BinaryMemebershipMatrix, *, n_clusters):
        model = rsnns.art2(
            np.asarray(bmm),
            f2Unit=n_clusters
        )

        print(model['fitted.values'])

if __name__ == '__main__':
    # rsnns = importr('RSNNS')
    # model = rsnns.art2(
    #     patterns,
    #     f2Unit=2#,
    #     #learnFuncParams=robjects.FloatVector([0.99, 20, 20, 0.1, 0]),
    #     #updateFuncParams=robjects.FloatVector([0.99, 20, 20, 0.1, 0])
    # )
    # predict = robjects.r('predict')
    # predictions = predict(model, testPatterns)
    pass
