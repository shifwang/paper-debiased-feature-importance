import sys
sys.path += ['../../unbiased-feature-importance']
import UFI
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
def UFI_importance(rf, X, y):
    if type(rf) == RandomForestRegressor:
        return UFI.regr(rf, X, y)
    elif type(rf) == RandomForestClassifier:
        return UFI.cls(rf, X, y)
    else:
        raise ValueError('type(rf) not recognized. {}'.format(type(rf)))
