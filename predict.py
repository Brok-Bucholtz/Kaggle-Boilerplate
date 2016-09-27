from argparse import ArgumentParser
from configparser import ConfigParser
from glob import glob
from os.path import exists, splitext

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

from model.model import classifiers
from process import process_data


def _get_data(files):
    """
    Get Dataframe of all the files
    :param files: File paths to get the data
    :return: Dataframe of all the data from the file paths
    """
    assert len(set([splitext(file)[1] for file in files])) == 1, 'All data files are not the same format'

    all_data = []
    extension = splitext(files[0])[1][1:]

    for file in files:
        if extension == 'csv':
            all_data.append(pd.read_csv(file))
        else:
            raise Exception('Extension not recognized')

    return pd.concat(all_data)


def _score(clf, features, labels):
    """
    Returns score for the prediction
    :param true_labels: The correct labels
    :param predict_labels: The prediction labels
    :return: The score of the prediction labels on the true labels
    """
    # BoilerPlate: Replace with the score function used in the Kaggle competition
    score = accuracy_score(labels, clf.predict(features))

    return score


def run():
    config = ConfigParser()
    config.read('config.ini')

    assert\
        len([
            value for section in config.sections()
            for key, value in config.items(section)
            if value == '<EDIT_ME>']) == 0,\
        'Missing information in config file'

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model', help='The model to run a prediction on', choices=['all'], default='all')
    arg_parser.add_argument('--create_submission', help='Create a submission', action='store_true')
    args = arg_parser.parse_args()

    test_features, train_features, test_labels, train_labels = train_test_split(
        *process_data(_get_data(glob(config['DIRECTORY']['Data'] + config['DATA']['TrainFiles']))),
        test_size=0.33)
    clfs = classifiers(args.model)

    for clf_data in clfs:
        grid_search = GridSearchCV(clf_data['model'](), clf_data['parameters'], scoring=_score)
        grid_search.fit(train_features, train_labels)
        predict_score = _score(grid_search, test_features, test_labels)

        print('----------------')
        print('Model {}:'.format(clf_data['name']))
        print('Parameters: {}'.format(grid_search.best_params_))
        print('Score: {}'.format(predict_score))

        if args.create_submission:
            file_path = config['DIRECTORY']['Submission'] + str(predict_score) + '--' + clf_data['name'] + '.csv'
            if not exists(file_path):
                submission = process_data(_get_data(glob(
                    config['DIRECTORY']['Data'] + config['DATA']['TestFiles'])),
                    train=False)

                assert isinstance(submission, pd.DataFrame), 'Data for submission must be a Data Frame'

                submission.to_csv(file_path)


if __name__ == '__main__':
    run()
