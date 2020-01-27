import numpy as np
import numba as nb
from sklearn import datasets
from scipy.spatial.distance import cdist as _cdist
from tqdm.auto import tqdm


@nb.njit
def sigmoid(x, b):
    return 1 / (1 + np.exp(-b*x))

@nb.njit
def similarity(e, data, data_sqrt):
    return np.dot(data, e)/(np.linalg.norm(e)*data_sqrt)

@nb.jit(nb.f8[:,:](nb.f8[:,:], nb.f8[:,:]), forceobj=True)
def cdist(a, b):
    return _cdist(a, b, 'euclidean')

def ant_cluster(data, *, n, m, s, v, v_max, a, b, space_scale=1.0, limit_plane=False, summary_freq=10):
    obj_positions = np.random.random([data.shape[0], 2]) * space_scale
    loaded = np.zeros(data.shape[:1], dtype=np.bool)
    loaded[:] = False
    loaded[np.random.choice(range(data.shape[0]), n, replace=False)] = True

    data_sqrt = np.sqrt(np.sum(data**2, axis=1))
    Y = cdist(obj_positions, obj_positions)
    np.fill_diagonal(Y, np.inf)
    
    for iteration_idx, iteration in enumerate(tqdm(range(m))):
        sim = np.apply_along_axis(similarity, 1, data, data, data_sqrt)
        d = 1 - sim

        v_iter = v()
        f_ = 1 - d/(a*(1 + ((v_iter - 1)/v_max)))
        f = np.sum(np.where(Y < s, f_, 0), axis=1) / s**2
        f[f <= 0] = 0

        drop_p = sigmoid(f, b)
        movement = np.random.random(obj_positions.shape) - 0.5
        movement *= v_iter

#         p = np.random.random(obj_positions.shape[0])
        p = np.random.random()

        loaded[drop_p > p] = False
        n_free_ants = n - np.count_nonzero(loaded)
        pickable_objs = np.flatnonzero(np.logical_and(loaded == False, drop_p <= p))
        picked_objs = np.random.choice(pickable_objs, min(n_free_ants, len(pickable_objs)), replace=False) if len(pickable_objs) else np.array([], dtype=bool)
        loaded[picked_objs] = True

        obj_positions[loaded] += movement[loaded]
        
        if limit_plane:
            obj_positions[loaded] = obj_positions[loaded].clip(0, space_scale)
            
        nY = cdist(obj_positions[loaded], obj_positions)
        Y[loaded, :] = nY
        Y[:, loaded] = nY.T
        np.fill_diagonal(Y, np.inf)
        
        if iteration_idx % (m // summary_freq) == 0 or summary_freq is None:
            yield np.copy(obj_positions), 1-drop_p
            
def create_plot_data(iter_data, target, pick_prob=False):
    for c_data, f in iter_data:
        c_data = np.append(c_data, target, axis=1)
        if pick_prob:
            c_data = np.append(c_data, f[:, None], axis=1)
        yield c_data
