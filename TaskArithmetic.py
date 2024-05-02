import argparse
import copy
import logging
import os
import time

import torch

from src.merging.utils.model_ops import *
from src.merging.utils.model_utils import *
from src.merging.utils.utils import *
from src.model.ModelConfig import ModelConfig
from src.model.utils import *
from src.train.TrainingConfig import TrainingConfig
from src.utils.config_utils import *
from src.utils.io import *
from src.utils.utils import *

from src.merging.methods.MergingMethod import MergingMethod


def compute_taskVectors(
    invidual_checkpoints: list[dict[str, torch.tensor]],
    pretrained_checkpoint: dict[str, torch.tensor],
):
    """
    Because various methods use task vectors, we make it a static function

    Args:
        task_checkpoints:
        pretrained_model:

    Returns:

    """

    taskVector_models = list(
        map(
            lambda checkpoint: elementWise_subtract(checkpoint, pretrained_checkpoint),
            invidual_checkpoints,
        )
    )
    return taskVector_models


class TaskArithmetic(MergingMethod):
    def __init__(
        self,
    ):
        super().__init__()

    def _checkMergingMethod_inConfig(self, merging_config):
        assert merging_config.method == "task_arithmetic"

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

    def get_taskVectors(
        self,
        invidual_checkpoints: list[dict[str, torch.tensor]],
        pretrained_checkpoint: dict[str, torch.tensor],
    ):
        """

        Args:
            task_checkpoints:
            pretrained_model:

        Returns:

        """
        return compute_taskVectors(invidual_checkpoints, pretrained_checkpoint)

    def task_arithmetic(self, checkpoints, pretrained_checkpoint, merging_lambda):

        taskVector_parameters = self.get_taskVectors(checkpoints, pretrained_checkpoint)

        summed_model = efficientReduceSum_modelParameters(taskVector_parameters)
        scaled_model = elementWise_scale(summed_model, merging_lambda)
        merged_model = elementWise_add(scaled_model, pretrained_checkpoint)

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
    ):
        """

        Args:
            merging_lambda:
            checkpoint_metadatas:
            parameter_name:
            pretrained_parameter:

        Returns:

        """
        self._checkMergingMethod_inConfig(merging_config)

        pretrained_parameter = load_pretrainedParameter(
            model_config,
            merging_config,
            finalParameter_name,
            original_parameterNames,
            device,
        )
        loaded_parameters = list(loaded_parameters.values())

        return (
            self.task_arithmetic(
                loaded_parameters, pretrained_parameter, merging_config.merging_lambda
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
        self._checkMergingMethod_inConfig(merging_config)
        pretrained_checkpoint = load_pretrainedCheckpoint(model_config, device)
        loaded_checkpoints = list(loaded_checkpoints.values())

        return (
            self.task_arithmetic(
                loaded_checkpoints, pretrained_checkpoint, merging_config.merging_lambda
            ),
            None,
        )
