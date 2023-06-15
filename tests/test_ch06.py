from pytest import *
from model import CountingDealingPartition


def s1():
    samples = [
        {
            "sepal_length": i + 0.4,
            "sepal_width": i + 0.2,
            "petal_length": i + 0.1,
            "petal_width": i + 0.2,
            "species": f"sample {i}"
        }
        for i in range(15)
    ]
    return samples


def test_dealingpartition(s1):
    dealing = CountingDealingPartition(s1)
    assert len(dealing.training) == 8
    assert len(dealing.testing) == 2

