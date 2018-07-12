# __Author__:Zcc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def frequency_cut(series, bin, return_point=True):
    """
    返回每个值所属的箱，如果return_point=True，返回每个切分的点
    由于样本的分布不同，可能会出现bin减少的情况(总箱数少于指定的bin),区间是前开后闭,箱数从1开始
    split_point = [2, 3, 5], 则区间bin为4个,(-inf, 2],(2,3],(3,5],(5, +inf]
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    edges = np.array([float(i) / bin for i in range(bin + 1)])

    # 返回第一个edges的rank大于实际rank的索引
    bin_result = np.array(series.rank(pct=1).apply(lambda x: (edges >= x).argmax()))
    split_point = [max(series[bin_result == point]) for point in np.unique(bin_result)[:-1]]
    if len(np.unique(bin_result)) < bin:
        bin_set = np.unique(bin_result)
        replace_dict = {bin_set[i]: i+1 for i in range(len(bin_set)) if bin_set[i] != i+1}
        for k, v in replace_dict.items():
            bin_result[bin_result == k] = v

    if return_point:
        return bin_result, split_point
    else:
        return bin_result


def chi2(group_1, group_2):
    """
    计算2个区间的卡方值, group1和2是某2个箱的所有值，里面元素是target,target_list
    group1 = [0,1,1,0] list
    group2 = [1,0,0,0] list
    return chi2
    """
    if not isinstance(group_1, np.ndarray):
        group_1 = np.array(group_1)
    if not isinstance(group_2, np.ndarray):
        group_2 = np.array(group_2)

    total = np.append(group_1, group_2)
    target_list = np.unique(total)

    a = []; e = []
    for y in target_list:
        y_rate = (total == y).sum()/total.shape[0]
        a.append(((group_1 == y).sum(), (group_2 == y).sum()))
        e.append(((group_1.shape[0]) * y_rate, group_2.shape[0] * y_rate))

    a = np.array(a)
    e = np.array(e)

    return ((a-e)**2/e).sum()


def calc_woe(column, target):
    if not isinstance(column, np.ndarray):
        column = np.array(column)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    good_total = (target == 1).sum()
    bad_total = (target == 0).sum()

    col_level = np.unique(column)
    woe_iv = {'woe': {}}
    iv_sum = 0
    for bin_ in col_level:
        good_bin = (target[column == bin_] == 1).sum()
        bad_bin = (target[column == bin_] == 0).sum()
        good_rate = good_bin/good_total
        bad_rate = bad_bin/bad_total
        woe_bin = np.log(good_rate/bad_rate)
        iv_bin = (good_rate - bad_rate) * woe_bin
        woe_iv['woe'][bin_] = woe_bin
        iv_sum += iv_bin
        woe_iv['iv'] = iv_sum
    return woe_iv


def prob_score(prob, basepoint, pdo):
    """将概率转化成分数且为正整数"""
    y = np.log(prob/(1-prob))
    return int(basepoint+pdo/np.log(2)*(-y))


def ks_score(y_true, y_predict, plot=False):
    """计算KS值"""
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_predict, np.ndarray):
        y_predict = np.array(y_predict)

    # 将y_predict 从高到低排序，返回索引
    ind = np.argsort(y_predict)[::-1]
    reverse = np.power(0, y_true)
    bad_rate_curve = y_true[ind].cumsum()/y_true.sum()
    good_rate_curve = reverse[ind].cumsum()/reverse.sum()
    if plot:
        plt.plot(range(len(y_true)), bad_rate_curve)
        plt.plot(range(len(y_true)), good_rate_curve)
        plt.show()
    return max(bad_rate_curve - good_rate_curve)


def convert_col_index(X, col_name):
    """将列名转换成列索引
    """
    col_index = list()
    for name in col_name:
        col_index.append(list(X.columns).index(name))
    return col_index












