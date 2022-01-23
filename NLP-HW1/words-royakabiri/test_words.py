import pytest

import numpy as np

import words


def test_most_common_tokens():
    assert words.most_common("brown_sample.txt", n=2) == [
        ('the/at', 5558),
        (',/,', 5133),
    ]


def test_most_common_words():
    assert words.most_common("brown_sample.txt",
                             word_regex=".*",
                             pos_regex=None,
                             n=10) == [
        ('the', 5580),
        (',', 5188),
        ('.', 4030),
        ('of', 2849),
        ('and', 2146),
        ('to', 2116),
        ('a', 1993),
        ('in', 1893),
        ('for', 943),
        ('The', 806),
    ]


def test_most_common_parts_of_speech():
    assert words.most_common("brown_sample.txt",
                             word_regex=None,
                             pos_regex=".*",
                             n=5) == [
        ('nn', 13162),
        ('in', 10616),
        ('at', 8893),
        ('np', 6866),
        (',', 5133),
    ]


def test_most_common_verbs():
    assert words.most_common("brown_sample.txt",
                             word_regex=".*",
                             pos_regex="vb.*",
                             n=3) == [
        ('said/vbd', 382),
        ('get/vb', 66),
        ('made/vbn', 62),
    ]


def test_most_common_ing_words():
    assert words.most_common("brown_sample.txt",
                             word_regex=".*ing$",
                             pos_regex=".*",
                             n=4) == [
        ('meeting/nn', 58),
        ('being/beg', 56),
        ('during/in', 55),
        ('including/in', 25),
    ]


def test_most_similar_to_house():
    wv = words.WordVectors("vectors_top3000.txt")
    assert wv.most_similar("house", 3) == [
        ('white', pytest.approx(0.609, abs=1e-3)),
        ('speaker', pytest.approx(0.549, abs=1e-3)),
        ('congressional', pytest.approx(0.537, abs=1e-3)),
    ]


def test_most_similar_to_white():
    wv = words.WordVectors("vectors_top3000.txt")
    assert wv.most_similar("white", 5) == [
        ('house', pytest.approx(0.609, abs=1e-3)),
        ('black', pytest.approx(0.567, abs=1e-3)),
        ('blue', pytest.approx(0.447, abs=1e-3)),
        ('speaker', pytest.approx(0.446, abs=1e-3)),
        ('pink', pytest.approx(0.445, abs=1e-3)),
    ]


def test_average():
    wv = words.WordVectors("vectors_top3000.txt")
    average_vector = wv.average_vector(["white", "house"])
    assert isinstance(average_vector, np.ndarray)
    assert average_vector[:5] == pytest.approx(np.array([
        -0.110, -0.018, 0.173, 0.055, 0.096]), abs=1e-3)
