import numpy as np
from scipy.spatial.distance import cdist

def ant_cluster(data, *, n, m, s, v, v_max, a, b):
    if not callable(v):
        v_ = lambda: v

    def sigmoid(x):                                        
        return 1 / (1 + np.exp(-b*x))
    
    obj_positions = np.random.random([data.shape[0], 2])

    for iteration in range(m):
        print(obj_positions)
        Y = cdist(obj_positions, obj_positions, 'euclidean')

        Y_sq = np.sqrt(np.sum(Y**2, axis=1))
        sim = np.apply_along_axis(lambda e: np.sum(e * Y, axis=1)/(np.sqrt(np.sum(e**2))*Y_sq), 1, Y)
        d = 1 - sim

        f_ = 1 - d/(a*(1 + ((v_() - 1)/v_max)))
        f = 1/s**2 * (np.sum(np.where(Y < s, f_, 0), axis=1) - 1)

        pd = sigmoid(f)
        pp = 1 - pd

        print(pp)


if __name__ == '__main__':
    np.random.seed(6)
    data = np.array(
        [
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
            np.random.randint(100, size=5),
        ]
    )
    ant_cluster(data, n=3, m=1, s=0.1, v=2, v_max=3, a=1, b=1)
