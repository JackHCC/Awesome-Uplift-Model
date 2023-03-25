import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Models.TreeModel.baseline import Baseline, PickleableMixin


class RandomForest(PickleableMixin, Baseline):
    def __init__(self):
        super(RandomForest, self).__init__()

    def _build(self, **kwargs):
        num_units = int(np.rint(kwargs["num_units"]))
        num_layers = int(np.rint(kwargs["num_layers"]))
        return RandomForestRegressor(n_estimators=num_units, max_depth=num_layers)

    def preprocess(self, x):
        return np.concatenate([x[0], np.atleast_2d(np.expand_dims(x[1], axis=-1))], axis=-1)

    def postprocess(self, y):
        if y.ndim > 1:
            return y[:, -1]
        else:
            return y
