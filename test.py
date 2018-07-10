# __Author__:Zcc
from chi_merge import ChiMerge
import time
import numpy as np

start_time = time.time()
# chi = ChiMerge(num_features=list(range(15)), cat_features=list(range(15, 30)))
#
# X = np.arange(6000000).reshape(200000, 30)
# y = np.random.randint(0, 2, size=200000)
#
# chi.fit(X, y)
# X_result = chi.transform(X)
# print(X_result)
#
time.sleep(1)
print(time.time()-start_time)

