import argparse
import copy
import logging
import os
import time

import torch

from src.utils.io import *
from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.model_utils import *
from src.model.ModelConfig import ModelConfig
from src.model.utils import *
from src.utils.config_utils import *


class MergingMethod(object):

    def _getMergingConfigs_withDifferentMergingLambda(
        self,
        merging_config: MergingConfig,
    ) -> List[MergingConfig]:
        """
        Get merging config with different hyperparameter if the hyperparameter is not fixed.

        Args:
            merging_config:

        Returns:
            all_mergingConfig
        """
        if merging_config.merging_lambda is None:
            all_mergingConfig = []
            for merging_lambda in iterate11Values_from0To1():
                all_mergingConfig.append(
                    update_mergingConfig(
                        merging_config, {"merging_lambda": merging_lambda}
                    )
                )
            return all_mergingConfig
        else:
            return [merging_config]

    def _getMergingConfigs_withDifferentDropoutProbability(
        self,
        merging_config: MergingConfig,
    ) -> List[MergingConfig]:
        """
        Get merging config with different hyperparameter if the hyperparameter is not fixed.

        Args:
            merging_config:

        Returns:
            all_mergingConfig
        """
        if merging_config.dropout_probability is None:
            all_mergingConfig = []
            for merging_lambda in iterate11Values_from0To1():
                all_mergingConfig.append(
                    update_mergingConfig(
                        merging_config, {"dropout_probability": merging_lambda}
                    )
                )
            return all_mergingConfig
        else:
            return [merging_config]

    def _getMergingConfigs_withDifferentNumberOfIterations(
        self,
        merging_config: MergingConfig,
    ) -> List[MergingConfig]:
        """
        Get merging config with different hyperparameter if the hyperparameter is not fixed.
        For averaging, the hyperparameter is merging_lambda

        Args:
            merging_config:

        Returns:
            all_mergingConfig
        """
        if merging_config.number_of_iterations is None:
            all_mergingConfig = []
            for number_of_iterations in iterate10Values_from10To100():
                all_mergingConfig.append(
                    update_mergingConfig(
                        merging_config, {"number_of_iterations": number_of_iterations}
                    )
                )
            return all_mergingConfig
        else:
            return [merging_config]

    def getMergingConfigs_withDifferentHyperparameters(
        self,
        merging_config: MergingConfig,
    ) -> List[MergingConfig]:
        raise NotImplementedError

    def requires_asynchronousStatistic(self):
        raise NotImplementedError

    def requires_synchronousStatistic(self):
        raise NotImplementedError

    def asynchronous_merge(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        loaded_parameters: list[Dict],
        loaded_parameterStatistics: list[Dict],
        finalParameter_name: str,
        original_parameterNames: list[str],
        device,
    ):
        raise NotImplementedError

    def synchronous_merge(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        loaded_checkpoints: Dict[str, Dict[str, torch.tensor]],
        loaded_statistics: Dict[str, Dict[str, torch.tensor]],
        device,
    ):
        raise NotImplementedError
