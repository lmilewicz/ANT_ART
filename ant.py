import numpy as np
import numba as nb
from sklearn import datasets
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
# import holoviews as hv
# from holoviews import dim
# import param
# import panel as pn
# from bokeh.plotting import show
# from bokeh.layouts import column
# hv.extension('bokeh')

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

        sim = np.apply_along_axis(lambda e: np.sum(e * data, axis=1)/(np.sqrt(np.sum(e**2))*data_sq), 1, data)
        d = 1 - sim

        f_ = 1 - d/(a*(1 + ((v_() - 1)/v_max)))
        f = np.maximum(1/s**2 * (np.sum(np.where(Y < s, f_, 0), axis=1) - 1), 0)

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

    # yield obj_positions, None,


# if __name__ == '__main__':
#     np.random.seed(6)
#     data = np.array(
#         [
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#             np.random.randint(100, size=5),
#         ]
#     )
#     iris = datasets.load_iris()
#     data = iris['data'][:, :2]
#     data = np.append(data, iris['target'][:, None]+1, axis=1)
#
#     iris_plt = hv.Scatter(data, vdims=['y', 'z'])
#     iris_plt.opts(color='z', cmap='Accent')
#     plots = []
#
#     v_max = 0.85
#     v = lambda: (np.random.random(data.shape[0]) * v_max)[:, np.newaxis]
#     for c_data in ant_cluster(data, n=50, m=100, s=3, v=v, v_max=v_max, a=1.5, b=3.0):
#         c_data = np.append(c_data, iris['target'][:, None]+1, axis=1)
#
#
#         clustered = hv.Scatter(c_data, vdims=['y', 'z'])
#         clustered.opts(color='z', cmap='Accent')
#         plots.append(clustered)
#
#     iteration = pn.widgets.IntSlider(name='Iteration', value=0, start=0, end=len(plots))
#
#     @pn.depends(iteration=iteration.param.value)
#     def load_symbol_cb(iteration):
#         return plots[iteration]
#
#     dmap = hv.DynamicMap(load_symbol_cb)
#     dashboard = pn.Row(pn.WidgetBox('## ANT', iteration))
#     # dashboard
#     # figs = [hv.render(p) for p in plots]
#     # show(column(*figs))
