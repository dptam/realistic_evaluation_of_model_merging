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
from src.train.TrainingConfig import TrainingConfig
from src.utils.config_utils import *

from src.merging.methods.MergingMethod import MergingMethod


class Average(MergingMethod):
    def __init__(
        self,
    ):
        super().__init__()

    def _checkMergingMethod_inConfig(self, merging_config):
        assert merging_config.method == "average"

    def getMergingConfigs_withDifferentHyperparameters(
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
        self._checkMergingMethod_inConfig(merging_config)
        return super()._getMergingConfigs_withDifferentMergingLambda(merging_config)

    def requires_asynchronousStatistic(self):
        return False

    def requires_synchronousStatistic(self):
        return False

    def average(
        self, checkpoints: List[Dict[str, torch.tensor]], merging_lambda: float
    ):
        # If merging 2 checkpoints, allow for a weighted average
        if len(checkpoints) == 2:
            scaled_checkpoints = [
                elementWise_scale(checkpoints[0], merging_lambda),
                elementWise_scale(checkpoints[1], (1 - merging_lambda)),
            ]
            checkpoints = scaled_checkpoints
            scaling_factor = 1
        else:
            assert merging_lambda == 1.0
            # Divide by number of checkpoints to get the average.
            scaling_factor = 1 / len(checkpoints)

        merged_model = elementWise_scaleAndSum(checkpoints, scaling_factor)

        return merged_model

    def asynchronous_merge(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        loaded_parameters: Dict[str, Dict[str, torch.tensor]],
        loaded_parameterStatistics: Dict[str, Dict[str, torch.tensor]],
        finalParameter_name: str,
        original_parameterNames: list[str],
        device,
    ) -> (Dict[str, torch.tensor], str | None):
        """
        Args:
            merging_lambda:
            loaded_checkpoints:

        Returns:
            model
        """
        self._checkMergingMethod_inConfig(merging_config)
        return (
            self.average(
                list(loaded_parameters.values()), merging_config.merging_lambda
            ),
            None,
        )

    def synchronous_merge(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        loaded_checkpoints: Dict[str, Dict[str, torch.tensor]],
        loaded_statistics: Dict[str, Dict[str, torch.tensor]],
        device,
    ):
        """
        Args:
            merging_lambda:
            loaded_checkpoints:

        Returns:
            model
        """
        self._checkMergingMethod_inConfig(merging_config)

        return (
            self.average(
                list(loaded_checkpoints.values()), merging_config.merging_lambda
            ),
            None,
        )
