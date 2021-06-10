import pytest

import numpy as np

import torch
import torch.nn as nn

from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10

from privacy_evaluator.metrics import compute_privacy_risk_score
from privacy_evaluator.classifiers.classifier import Classifier


def test_privacy_risk_score():
    x_train, y_train, x_test, y_test = CIFAR10.numpy()
    classifier = Classifier(
        load_dcti(),
        nb_classes=CIFAR10.N_CLASSES,
        input_shape=CIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )
    score = compute_privacy_risk_score(
        classifier, x_train[:100], y_train[:100], x_test[:100], y_test[:100]
    )
    assert bool(score)
