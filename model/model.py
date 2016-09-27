from sklearn.ensemble import RandomForestClassifier


def classifiers(model):
    clfs = {
        'random_forest': {
            'name': 'sklearn-random_forest_classifier',
            'model':  RandomForestClassifier,
            'parameters': {}}
    }

    if model == 'all':
        return clfs.values()

    return [clfs[model]]
