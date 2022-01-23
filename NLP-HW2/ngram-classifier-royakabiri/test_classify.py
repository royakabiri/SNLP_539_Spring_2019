import numpy as np
import pytest
from sklearn.metrics import f1_score, accuracy_score

import classify


@pytest.fixture(autouse=True)
def set_seeds():
    np.random.seed(42)


def test_read_smsspam():
    # keep a counter here (instead of enumerate) in case the iterator is empty
    count = 0
    for example in classify.read_smsspam("smsspam/SMSSpamCollection.train"):

        # make sure the right shape is returned
        assert len(example) == 2
        label, text = example

        # make sure the label is one of the expected two
        assert label in {"ham", "spam"}

        count += 1
    assert count == 3345


def test_features():
    # get the texts from the training data
    examples = classify.read_smsspam("smsspam/SMSSpamCollection.train")
    texts = [text for _, text in examples]

    # create the feature extractor from the training texts
    to_features = classify.TextToFeatures(texts)

    # extract features for some made-up sentences
    features = to_features(["There are some things that I need to send to you.",
                            "Hello!"])

    # make sure there is one row of features for each sentence
    assert len(features.shape) == 2
    n_rows, n_cols = features.shape
    assert n_rows == 2

    # make sure there are nonzero values for some selected unigram and bigram
    # features in the first sentence
    indices = [to_features.index(f) for f in ["need", "to you"]]
    assert len(set(indices)) > 1
    row_indices, col_indices = features[:, indices].nonzero()
    assert np.all(row_indices == 0)
    assert len(col_indices) == 2


def test_labels():
    # get the texts from the training data
    examples = classify.read_smsspam("smsspam/SMSSpamCollection.train")
    labels = [label for label, _ in examples]

    # create the label encoder from the training texts
    to_labels = classify.TextToLabels(labels)

    # make sure that some sample labels are encoded as expected
    ham_index = to_labels.index("ham")
    spam_index = to_labels.index("spam")
    assert ham_index != spam_index
    assert np.all(to_labels(["ham", "spam", "spam"]) ==
                  [ham_index, spam_index, spam_index])


def test_prediction(capsys, min_f1=0.89, min_accuracy=0.97):
    # get texts and labels from the training data
    train_examples = classify.read_smsspam("smsspam/SMSSpamCollection.train")
    train_labels, train_texts = zip(*train_examples)

    # get texts and labels from the development data
    devel_examples = classify.read_smsspam("smsspam/SMSSpamCollection.devel")
    devel_labels, devel_texts = zip(*devel_examples)

    # create the feature extractor and label encoder
    to_features = classify.TextToFeatures(train_texts)
    to_labels = classify.TextToLabels(train_labels)

    # train the classifier on the training data
    classifier = classify.Classifier()
    classifier.train(to_features(train_texts), to_labels(train_labels))

    # make predictions on the development data
    predicted_indices = classifier.predict(to_features(devel_texts))
    assert np.array_equal(predicted_indices, predicted_indices.astype(bool))

    # measure performance of predictions
    devel_indices = to_labels(devel_labels)
    spam_label = to_labels.index("spam")
    f1 = f1_score(devel_indices, predicted_indices, pos_label=spam_label)
    accuracy = accuracy_score(devel_indices, predicted_indices)

    # print out performance
    if capsys is not None:
        with capsys.disabled():
            msg = "\n{:.1%} F1 and {:.1%} accuracy on SMSSpam development data"
            print(msg.format(f1, accuracy))

    # make sure that performance is adequate
    assert f1 > min_f1
    assert accuracy > min_accuracy


@pytest.mark.xfail
def test_very_accurate_prediction():
    test_prediction(capsys=None, min_f1=0.94, min_accuracy=0.98)
