import numpy as np
from sklearn import datasets
from scipy.spatial.distance import cdist
from tqdm import tqdm
import holoviews as hv
from holoviews import dim
from bokeh.plotting import show
from bokeh.layouts import row
hv.extension('bokeh')

def ant_cluster(data, *, n, m, s, v, v_max, a, b):
    if not callable(v):
        v_ = lambda: v

    def sigmoid(x):
        return 1 / (1 + np.exp(-b*x))

    obj_positions = np.random.random([data.shape[0], 2])
    loaded = np.zeros(data.shape[:1], dtype=np.bool)
    loaded[:n] = True

    for iteration in tqdm(range(m)):
        # print(obj_positions)
        Y = cdist(obj_positions, obj_positions, 'euclidean')

        data_sq = np.sqrt(np.sum(data**2, axis=1))
        sim = np.apply_along_axis(lambda e: np.sum(e * data, axis=1)/(np.sqrt(np.sum(e**2))*data_sq), 1, data)
        d = 1 - sim

        f_ = 1 - d/(a*(1 + ((v_() - 1)/v_max)))
        f = 1/s**2 * (np.sum(np.where(Y < s, f_, 0), axis=1) - 1)

        drop_p = sigmoid(f)
        movement = v_() * (np.random.random(obj_positions.shape) - 0.5)

        # np.random.seed(3)
        p = np.random.random()

        loaded[drop_p < p] = False
        n_free_ants = n - np.count_nonzero(np.logical_and(drop_p < p, loaded))
        pickable_objs = np.flatnonzero(np.logical_and(loaded == False, drop_p >= p))
        picked_objs = np.unique(np.random.choice(pickable_objs, n_free_ants) if len(pickable_objs) else np.array([], dtype=bool))
        loaded[picked_objs] = True

        # print(n_free_ants)
        # print(loaded)
        # print(movement)
        # print(picked_objs)

        obj_positions[loaded] += movement[loaded]

    return obj_positions


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
    iris = datasets.load_iris()
    data = iris['data'][:, :2]
    data = np.append(data, iris['target'][:, None]+1, axis=1)
    c_data = ant_cluster(data, n=50, m=10000, s=0.1, v=1, v_max=1, a=0.8, b=0.9)
    c_data = np.append(c_data, iris['target'][:, None]+1, axis=1)

    iris = hv.Scatter(data, vdims=['y', 'z'])
    iris.opts(color='z')
    clustered = hv.Scatter(c_data, vdims=['y', 'z'])
    clustered.opts(color='z')
    plots = [iris, clustered]
    figs = [hv.render(p) for p in plots]
    show(row(*figs))
