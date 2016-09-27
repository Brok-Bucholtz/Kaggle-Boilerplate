from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def classifiers(model):
    clfs = {
        'random_forest': {
            'name': 'sklearn-random_forest_classifier',
            'model':  RandomForestClassifier,
            'parameters': {}},
        'gradient_boosting': {
            'name': 'xgboost-xgb_classifier',
            'model': XGBClassifier,
            'parameters': {}}}

    if model == 'all':
        return clfs.values()

    return [clfs[model]]
