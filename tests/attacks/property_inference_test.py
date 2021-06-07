<<<<<<< HEAD
from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.tf.cnn import ConvNet


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader("CIFAR10")
    input_shape = test_dataset[0][0].shape
    num_elements_per_classes = {0: 1000, 1: 1000}
    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)

    model = ConvNet(num_classes, input_shape)
    trainer(train_set, num_elements_per_classes, model)
=======
import pytest

from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.models.train_cifar10_torch.data import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.models.train_cifar10_torch.train import trainer_out_model


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader()
    input_shape = [32, 32, 3]
    num_classes = 2
    num_elements_per_classes = {0: 5000, 1: 5000}

    train_set, test_set = new_dataset_from_size_dict(
        train_dataset, test_dataset, num_elements_per_classes
    )

    accuracy, model = trainer_out_model(
        train_set, test_set, num_elements_per_classes, "FCNeuralNet"
    )
>>>>>>> Team2sprint2 (#102)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)

<<<<<<< HEAD
    attack = PropertyInferenceAttack(target_model, train_dataset)
=======
    attack = PropertyInferenceAttack(target_model)
>>>>>>> Team2sprint2 (#102)
    attack.attack()
