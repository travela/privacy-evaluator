from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
import numpy as np

import os
import torch
import torchvision
from typing import Tuple, Dict, Union, Optional


class PropertyInferenceAttackSkeleton(PropertyInferenceAttack):
    def __init__(
            self,
            model,
            property_shadow_training_sets,
            negation_property_shadow_training_sets,
    ):
        """
        Initialize the Property Inference Attack Class.
        :param model: the target model to be attacked
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
        :param property_shadow_training_sets: the shadow training sets that fulfill property
        :type property_shadow_training_sets: np.ndarray # TODO
        :param negation_property_shadow_training_sets: the shadow training sets that fulfill negation of property
        :type negation_property_shadow_training_sets: np.ndarray # TODO
        """

        super().__init__(
            model, property_shadow_training_sets, negation_property_shadow_training_sets
        )
        # TODO: create shadow_training_set
        shadow_training_set = None
        self.shadow_training_set = shadow_training_set

    def dataset_downloader(
            dataset_name: str = "CIFAR10",
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Download the corresponding dataset, skip if already downloaded.
        Args:
            dataset_name: Name of the dataset.\n
        Returns:
            Train and test dataset, both of type `torch.utils.data.Dataset`.
        """
        dir_path = os.path.dirname(os.path.abspath(__file__))

        if dataset_name == "CIFAR10":
            # check if already downloaded
            data_path = os.path.join(dir_path, "../../../", dataset_name)
            downloaded = os.path.exists(os.path.join(data_path, "cifar-10-python.tar.gz"))
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=not downloaded,
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=not downloaded,
            )
        return train_dataset, test_dataset

    def subset(
            dataset: torch.utils.data.Dataset,
            class_id: int = 0,
            num_samples: Optional[int] = None,
    ) -> torch.utils.data.Dataset:
        """
        Take a subset from the whole dataset.
        First, fetch all the data points of the given `class_id` as subset,
        then randomly select `num_samples` many points from this class.
        Args:
            dataset: The dataset, usually containing multiple classes.
            class_id: The id for the target class we want to filter.
            num_samples: Sampling size of the class `class_id`.
        Returns:
            A subset from `dataset` with samples all in class `class_id` and of
            size `num_samples`. If `num_samples` is not specified, then keep all
            the samples in this class, which is the usual practice for test set.
        """
        idx = torch.tensor(dataset.targets) == class_id
        subset = torch.utils.data.dataset.Subset(
            dataset=dataset, indices=np.where(idx == True)[0]
        )

        if num_samples:
            assert num_samples <= len(subset)
            idx = np.random.choice(len(subset), num_samples, replace=False)
            subset = torch.utils.data.dataset.Subset(dataset=subset, indices=idx)
        return subset

    def create_shadow_training_set(self, dataset, amount_sets, size_set, class_ids):
        """
        # TODO: take test_dataset from dataset_downloader as input?
        :param data_set: np.ndarray
        :param amount_sets: int
        :return:
        """
        amount_property = int(round(amount_sets/2))
        amount_neg_property = int(round(amount_sets/2))

        size_dict = []


        list_property = []
        list_neg_property = []

        for i in range(amount_property):
            for j in class_ids
                shadow_training_set = subset(dataset, j, size)

            list_property.append(shadow_training_set)

        subsets = []
        for _, (class_id, size) in enumerate(size_dict.items()):
            subsets.append(subset(train_dataset, class_id, size))
        new_train_dataset = torch.utils.data.ConcatDataset(subsets)


    def train_shadow_classifiers(self, shadow_training_set):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_set: datasets used for shadow_classifiers
        :type shadow_training_set: np.ndarray # TODO
        :return: shadow classifiers
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`)
        """
        raise NotImplementedError

    def feature_extraction(self, model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
        :return: feature extraction
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def create_meta_training_set(self, feature_extraction_list):
        """
        Create meta training set out of the feature extraction of the shadow classifiers.
        :param feature_extraction_list: list of all feature extractions of all shadow classifiers
        :type feature_extraction_list: np.ndarray
        :return: Meta-training set
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def train_meta_classifier(self, meta_training_set):
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_set: Set of feature representation of each shadow classifier,
        labeled according to whether property or negotiation of property is fulfilled.
        :type meta_training_set: np.ndarray
        :return: Meta classifier
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`) # TODO only binary classifiers - special classifier?
        """
        raise NotImplementedError

    def perform_prediction(self, meta_classifier, feature_extraction_target_model):
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs property prediction.
        :param meta_classifier: A classifier
        :type meta_classifier: "CLASSIFIER_TYPE" (to be found in `.art.utils`)
        # TODO only binary classifiers-special classifier?
        :param feature_extraction_target_model: extracted features of target model
        :type feature_extraction_target_model: np.ndarray
        :return: Prediction whether property or negation of property is fulfilled for target data set
        :rtype: # TODO
        """
        raise NotImplementedError

    def perform_attack(self, params):
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set
        :rtype: # TODO
        """
        shadow_classifier = self.train_shadow_classifiers(self.shadow_training_set)
        # TODO: create feature extraction for all shadow classifiers
        feature_extraction_list = None
        meta_training_set = self.create_meta_training_set(feature_extraction_list)
        meta_classifier = self.train_meta_classifier(meta_training_set)
        # TODO: create feature extraction for target model, using x
        feature_extraction_target_model = None
        prediction = self.perform_prediction(
            meta_classifier, feature_extraction_target_model
        )
        return prediction