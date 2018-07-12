# __Author__:Zcc
from chi_merge import ChiMerge
from util import convert_col_index
import pandas as pd

X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv', header=None)
num_features = ['int_rate', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app', 'inq_last_6mths',
                'mths_since_last_record', 'mths_since_last_delinq', 'open_acc', 'pub_rec', 'total_acc', 'limit_income']

cat_features = ['home_ownership', 'verification_status', 'desc', 'purpose', 'zip_code', 'addr_state', 'pub_rec_bankruptcies']
num_feature_index = convert_col_index(X, num_features)
cat_features_index = convert_col_index(X, cat_features)
cm = ChiMerge(num_features=num_feature_index, cat_features=cat_features_index, special_value=-1, max_bin=20)

cm.fit(X, y)
X_merge = cm.transform(X)










