# __Author__:Zcc

# __Author__:Zcc

import numpy as np


class BaseEstimator(object):
    X = None
    y = None
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """校验格式
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError('数据集大小不能为0')

        if X.ndim == 1:
            self.n_samples, self.n_features = 1,X.shape[0]
        else:
            self.n_samples, self.n_features = X.shape[0],X.shape[1]

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError('缺少参数y')

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError('目标变量个数必须大于0')

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)




