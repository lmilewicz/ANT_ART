import numpy as np
from sklearn import datasets
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


def ant_cluster(data, *, n, m, s, v, v_max, a, b, space_scale=1.0):
    if not callable(v):
        v_ = lambda: v
    else:
        v_ = v

    def sigmoid(x):
        return 1 / (1 + np.exp(-b*x))

    obj_positions = np.random.random([data.shape[0], 2]) * space_scale
    loaded = np.zeros(data.shape[:1], dtype=np.bool)
    loaded[:] = False
    loaded[np.random.choice(range(data.shape[0]), n, replace=False)] = True

    data_sq = np.sqrt(np.sum(data**2, axis=1))
    for iteration_idx, iteration in enumerate(tqdm(range(m))):
        stats = {'iteration': iteration_idx}
        Y = cdist(obj_positions, obj_positions, 'euclidean')
        np.fill_diagonal(Y, np.inf)

        sim = np.apply_along_axis(lambda e: np.sum(e * data, axis=1)/(np.sqrt(np.sum(e**2))*data_sq), 1, data)
        d = 1 - sim

        f_ = 1 - d/(a*(1 + ((v_() - 1)/v_max)))
        f = np.maximum(1/s**2 * np.sum(np.where(Y < s, f_, 0), axis=1), 0)

        drop_p = sigmoid(f)
        movement = np.random.random(obj_positions.shape) - 0.5
        movement = movement / np.linalg.norm(movement, axis=0)
        movement *= v_()

        # np.random.seed(3)
        p = np.random.random(obj_positions.shape[0])
        # p = np.random.random()

        stats['dropped'] = np.count_nonzero(np.logical_and(loaded, drop_p > p))
        loaded[drop_p > p] = False
        n_free_ants = n - np.count_nonzero(loaded)
        stats['free ants'] = n_free_ants
        pickable_objs = np.flatnonzero(np.logical_and(loaded == False, drop_p <= p))
        stats['pickable'] = len(pickable_objs)
        picked_objs = np.random.choice(pickable_objs, min(n_free_ants, len(pickable_objs)), replace=False) if len(pickable_objs) else np.array([], dtype=bool)
        stats['picked'] = len(picked_objs)
        loaded[picked_objs] = True
        stats['loaded'] = np.count_nonzero(loaded)

        # print(n_free_ants)
        # print(loaded)
        # print(movement)
        # print(picked_objs)

        stats_str = ' | '.join(['{0}: {1}'.format(k, v) for k, v in stats.items()])
        # tqdm.write(stats_str)

        obj_positions[loaded] += movement[loaded]
        if iteration_idx % (m // 10) == 0 or True:
            yield obj_positions, stats, 1-drop_p
