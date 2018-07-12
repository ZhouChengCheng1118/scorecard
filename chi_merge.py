# __Author__:Zcc
import numpy as np
from base import BaseEstimator
from util import frequency_cut, chi2


class ChiMerge(BaseEstimator):

    def __init__(self, num_features, cat_features, max_bin=5, category_num=5, min_bin_pnt=0.1, special_value=None,
                 monotone_col=None, bad_target=1, init_bins=100):
        self.max_bin = max_bin
        self.category_num = category_num
        self.num_features = num_features  # 有序变量从1开始编码，当连续型变量处理
        self.cat_features = cat_features
        self.more_value_features = []
        self.less_value_features = []
        self.split_point = {'less_value_features': {}, 'more_value_features': {}, 'num_features': {}}
        self.min_bin_pnt = min_bin_pnt
        self.special_value = special_value
        self.Monotone_col = monotone_col  # 检查单调性的列
        self.bad_target = bad_target
        self.init_bins = init_bins

    def _check_category(self):
        """
        检查类别型变量中，哪些变量取值超过category_num, 取值超过5的变量, 需要bad rate编码, 再用卡方分箱法进行分箱.
        """
        if self.category_num < 0:
            raise ValueError("category_num should be more than 0")

        for var in self.cat_features:
            value_counts = len(np.unique(self.X[:, var]))
            if value_counts > self.category_num:
                self.more_value_features.append(var)
            else:
                self.less_value_features.append(var)

    @staticmethod
    def _bin_bad_rate(col, y, bad_target=1, special_value=None):
        """
        calculate bad rate for given a column.

        Parameters
        ----------
        col : array-like
        y : array-like
            target in the data
        bad_target : int, default 1
            bad sample target
        special_value : int, float or list
            the value do not need to be processed
        Returns
        -------
        result : dict
        """
        result = {}
        for value in np.unique(col):
            if np.isin(value, special_value):  # 如果有缺失值，则不放入字典
                continue
            tmp_y = y[col == value]
            result[value] = (tmp_y == bad_target).sum()/tmp_y.shape[0]
        return result

    def _merge_bad(self):
        # 把self.X 里类别特征取值少的，只有一种好坏样本的取值合并,依据bad_rate合并
        # return 返回所有类别特征取值少的特征合并后的列
        for column_index in self.less_value_features:
            tmp_dict = self._bin_bad_rate(self.X[:, column_index], self.y, bad_target=self.bad_target,
                                          special_value=self.special_value)
            if max(tmp_dict.values()) == 1. or min(tmp_dict.values()) == 0.:
                tmp_sort = sorted([(key, tmp_dict[key]) for key, value in tmp_dict.items() if value not in (1., 0.)],
                                  key=lambda x: x[1])
                tmp_min = tmp_sort[0][0]
                tmp_max = tmp_sort[-1][0]

                tmp_0 = {key: tmp_min for key, value in tmp_dict.items() if value == 0.}
                tmp_1 = {key: tmp_max for key, value in tmp_dict.items() if value == 1.}
                tmp_0.update(tmp_1)
                self.split_point['less_value_features'][column_index] = tmp_0  # less_feature中有改动的变量值，可能为空, 不含特殊值

                # self._dict_replace(self.X[:, column_index], tmp_0)  # 可能不是从1开始的编码
            else:
                self.split_point['less_value_features'][column_index] = {}  # 给定一个空字典
            # 如果最小编码不是从1开始，则重新排序
            # col_level_total = np.unique(self.X[:, column])
            # col_level_filter = col_level_total[~np.isin(col_level_total, self.special_value)]
            # if col_level_filter[0] != 1:  # 除特殊值之外最小值不为1，说明不是从1开始编码
            #     replace_dict = {value: index + 1 for index, value in enumerate(col_level_filter)}  # 该字典里没有特殊值
            #     self._dict_replace(self.X[:, column], replace_dict)

    # 当取值>5时：用bad rate进行编码，放入连续型变量里,缺失值不需要bad_rate编码，交给连续型变量处理
    def _badrate_encoding(self):
        for column_index in self.more_value_features:
            tmp_dict = self._bin_bad_rate(self.X[:, column_index], self.y, bad_target=self.bad_target,
                                          special_value=self.special_value)
            # self._dict_replace(self.X[:, column_index], tmp_dict)
            self.split_point['more_value_features'][column_index] = tmp_dict  # 暂时以 value：bad_rate代替，不含缺失值
            self.num_features.append(column_index)

    # 对某列连续性变量分箱
    def _continuous_merge(self, col_index, special_value=None, bad_target=1, max_bins=5, min_bin_pnt=0.1, init_bins=100,
                          monotone=True):
        """
        calculate bad rate for given a column.

        Parameters
        ----------
        col_index : int
        special_value : int, float or list, default None
            values is not taken into the calculation
        bad_target : int, default 1
            bad sample target
        max_bins : int, default 5
            maximum number of bins
        min_bin_pnt ：int, default 0
            the percent of the minmum box
        init_bins : int, default 100
            the number of initial bin, shoule be more than 5
        Returns
        -------
        result : list
            cut off points
        """
        special_bool = np.isin(self.X[:, col_index], special_value)
        tmp_col = self.X[:, col_index][~special_bool].copy()
        if col_index in self.more_value_features:  # 多分类变量先进行bad_rate编码
            tmp_dict = self.split_point['more_value_features'][col_index]
            self._dict_replace(tmp_col, tmp_dict)

        tmp_target = self.y[~special_bool]
        col_levels = np.unique(tmp_col)
        col_counts = len(col_levels)

        if col_counts >= init_bins:  # 等频划分,返回的tmp_col从1开始
            tmp_col, split_point = frequency_cut(tmp_col, init_bins)

        else:  # 每个值为一个组
            split_point = list(col_levels)[:-1]
            replace_dict = {col_levels[i]: i + 1 for i in range(col_counts)}
            self._dict_replace(tmp_col, replace_dict)  # tmp_col为从1开始的组
            if tmp_col.dtype == float:
                tmp_col = tmp_col.astype(int)

        # 求出每组的总样本数与坏样本数
        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        while len(split_point)+1 > max_bins:  # 当前箱数大于预设的箱数
            chi = []
            for index in range(1, len(split_point)+1):
                chi.append(chi2(tmp_target[tmp_col == index], tmp_target[tmp_col == index+1]))
            combine_bin = chi.index(min(chi))+1  # 合并的组（combin_bin, combin_bin+1）
            # tmp_col从1开始重新编码,split_point更新
            tmp_col, split_point = self._combine(combine_bin, tmp_col, split_point)

        # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本,依据卡方值最小合并

        tmp_col, split_point = self._check_bad_rate(tmp_col, tmp_target, split_point, bad_target=bad_target)

        # 需要检查分箱后的最小占比
        tmp_col, split_point = self._check_min_bin(tmp_col, tmp_target, split_point, min_bin_pnt=min_bin_pnt)

        # 检查该列分箱后bad_rate是否单调,选择性检查,不单调就减少max_bin再分箱
        if monotone:
            if not self._check_monotone(tmp_col, tmp_target, bad_target=bad_target):
                self._continuous_merge(col_index=col_index, special_value=special_value, bad_target=bad_target,
                                       max_bins=max_bins-1, min_bin_pnt=min_bin_pnt, init_bins=init_bins, monotone=monotone)

        # 不需要处理特殊值，因为该方法只需返回split_point
        # # 增加特殊值的箱,检查该列是否有特殊值
        # if special_bool.any():
        #     # 找出special_bool为True的全部索引
        #     replace_special = [(index, self.X[:, col_index][index]) for index in np.where(special_bool)[0]]
        #     for index, value in replace_special:
        #         tmp_col = np.insert(tmp_col, index, value)
        #     return tmp_col, split_point
        # else:
        return split_point

    def _check_monotone(self, column, y, bad_target=1):
        """
        :return True 单调，False不单调
        """
        if max(column) == 2:
            return True
        else:
            value_bad_rate = self._bin_bad_rate(column, y, bad_target=bad_target)

            for key in range(2, max(column)):
                cond1 = value_bad_rate[key] < value_bad_rate[key+1] and value_bad_rate[key] < value_bad_rate[key-1]
                cond2 = value_bad_rate[key] > value_bad_rate[key+1] and value_bad_rate[key] > value_bad_rate[key-1]
                if cond1 or cond2:
                    return False
            return True

    def _check_min_bin(self, column, y, split_point, min_bin_pnt):
        """
        检查要求的最小箱比例是否小于实际最小箱比例，合并
        """
        v, c = np.unique(column, return_counts=True)
        min_bin = v[c.argmin()]  # 占比最小的箱
        min_bin_act_pct = min(c / column.shape[0])
        if min_bin_pnt > min_bin_act_pct:

            if len(split_point) == 1:  # 只有2组时不需要合并
                return column, split_point
            elif min_bin == 1:
                # 占比最小的箱是第一箱，第一箱和第二箱合并
                col, split_point = self._combine(1, column, split_point)
            elif min_bin == max(column):
                # 最后一箱和前一箱合并
                col, split_point = self._combine(max(column) - 1, column, split_point)
            else:
                # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                if chi2(y[column == min_bin], y[column == min_bin - 1]) <= chi2(y[column == min_bin], y[column == min_bin + 1]):
                    col, split_point = self._combine(min_bin - 1, column, split_point)
                else:
                    col, split_point = self._combine(min_bin, column, split_point)
            # 递归
            self._check_min_bin(col, y, split_point, min_bin_pnt)
        return column, split_point

    def _check_bad_rate(self, column, y, split_point, bad_target=1):
        """返回tmp_col和split_point"""
        value_bad_rate = self._bin_bad_rate(column, y, bad_target=bad_target)
        if max(value_bad_rate.values()) == 1. or min(value_bad_rate.values()) == 0.:
            if len(split_point) == 1:  # 只有2组时该分箱完全区分target，需要检查特征
                raise ValueError('feature error')
                # 递归检查
            elif np.isin(value_bad_rate[1], [1., 0.]):
                # 第一箱的bad_rate为1或0和第二箱合并
                col, split_point = self._combine(1, column, split_point)
            elif np.isin(value_bad_rate[max(column)], [1., 0.]):
                # 最后一箱和前一箱合并
                col, split_point = self._combine(max(column)-1, column, split_point)
            else:
                # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                # 找出中间的哪一箱为bad_rate为0,箱数以第一次输入的col和split_point为准
                # 找到中间第一个bad_rate为0的箱 tmp_bin
                tmp_bin = [k for k, v in value_bad_rate.items() if v in [0., 1.]][0]
                if chi2(y[column == tmp_bin], y[column == tmp_bin-1]) <= chi2(y[column == tmp_bin], y[column == tmp_bin+1]):
                    col, split_point = self._combine(tmp_bin-1, column, split_point)
                else:
                    col, split_point = self._combine(tmp_bin, column, split_point)
            # 递归
            self._check_bad_rate(col, y, split_point=split_point, bad_target=bad_target)
        return column, split_point

    def _combine(self, combine_bin, column, split_point):
        """
        合并combine_bin 和 combine_bin+1组，返回更新后的col和split_point
        """
        if combine_bin == max(column):
            raise ValueError('combine_bin equal to max bin')
        # 确保合并和组别有序排列
        combine_replace = {bin_count: bin_count - 1 for bin_count in range(combine_bin + 1, len(split_point) + 2)}
        self._dict_replace(column, combine_replace)  # 更新tmp_col
        split_point.pop(combine_bin - 1)
        return column, split_point

    @staticmethod
    def _dict_replace(column, k_v):
        """
        根据k-v来替换col中的value,缺失值不需要放入
        """
        col = column.copy()
        for k, v in k_v.items():
            column[col == k] = v

    # 对连续型变量进行分箱，包括（ii）中的变量
    def _train_bin(self):
        num_features = self.num_features.copy()
        for ix in num_features:
            split_point = self._continuous_merge(col_index=ix, special_value=self.special_value, bad_target=self.bad_target,
                                                 max_bins=self.max_bin, min_bin_pnt=self.min_bin_pnt, init_bins=self.init_bins,
                                                 monotone=False if self.Monotone_col is None else ix in self.Monotone_col)
            if ix in self.more_value_features:
                # 更新tmp_dict, self.split_point['more_value_features'].update({column_index: tmp_dict}) tmp_dict为value：bad_rate
                for k, v in self.split_point['more_value_features'][ix].items():
                    self.split_point['more_value_features'][ix][k] = self._convert_bin(v, split_point)  # 把多变量的值从原始值映射到箱数

                self.num_features.remove(ix)
            else:
                self.split_point['num_features'][ix] = split_point

    @staticmethod
    def _convert_bin(value, split_point):
        """
        给定值返回对应的箱
        """
        if not isinstance(split_point, np.ndarray):
            split_point = np.array(split_point)
        if value > split_point[-1]:
            return len(split_point) + 1
        else:
            return np.searchsorted(split_point, value) + 1

    def _transform_continuous(self, continuous_col, split_point, special_value=None):
        """
        根据给定的某列连续值和切分点，返回值对应的箱，如果值中有特殊值，则不处理
        """
        special_bool = np.isin(continuous_col, special_value)
        col_level = np.unique(continuous_col[~special_bool])

        replace_dict = {value: self._convert_bin(value, split_point) for value in col_level}
        self._dict_replace(continuous_col, replace_dict)

    def _transform(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        for column_index in self.less_value_features:
            self._dict_replace(X[:, column_index], self.split_point['less_value_features'][column_index])

        for column_index in self.more_value_features:
            self._dict_replace(X[:, column_index], self.split_point['more_value_features'][column_index])

        for column_index in self.num_features:
            self._transform_continuous(X[:, column_index], self.split_point['num_features'][column_index],
                                       special_value=self.special_value)
        return X

    def transform(self, X):
        return self._transform(X)

    def fit(self, X, y=None):  # fit时不需要更改transform
        # 校验格式
        self._setup_input(X, y)
        # 区分出类别变量的取值
        self._check_category()
        # 少取值<5时：只有一类的好坏样本合并
        self._merge_bad()
        # 当取值>5时：用bad rate进行编码，放入连续型变量里
        self._badrate_encoding()
        # 对连续性变量卡方分箱
        self._train_bin()








































