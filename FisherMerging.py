import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.merging.save_statistic import *
from src.data.DatasetConfig import DatasetConfig
from src.eval.EvaluationConfig import EvaluationConfig
from src.merging.methods.MergingMethod import MergingMethod
from src.merging.utils.checkpoints import *
from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.model.ModelConfig import ModelConfig
from src.model.utils import *
from src.train.TrainingConfig import TrainingConfig


class FisherMerging(MergingMethod):
    def __init__(
        self,
    ):
        super().__init__()

    def _checkMergingMethod_inConfig(self, merging_config):
        assert merging_config.method == "fisher_merging"

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
        return True

    def requires_synchronousStatistic(self):
        return True

    def load_parameterStatistic(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        datasets: list[str],
        checkpoint_metadatas: Dict[str, Dict],
        finalParameter_name: str,
        original_parameterNames: list[str],
        device,
    ):
        assert merging_config.fisher_approximation == "diagonal"

        checkpointStatistics_metadatas = []
        for metadata in checkpoint_metadatas:
            new_metadata = {
                "checkpoint_dir": getCheckpointFisher_name(
                    metadata["checkpoint_dir"], merging_config
                ),
                "parameter_names": metadata["parameter_names"],
            }
            checkpointStatistics_metadatas.append(new_metadata)

        loaded_parameterStatistics = loadParameterOrStatistic_fromMetadatas(
            model_config,
            merging_config,
            datasets,
            checkpointStatistics_metadatas,
            finalParameter_name,
            original_parameterNames,
            "statistics",
            device,
        )
        return loaded_parameterStatistics

    def load_checkpointStatistic(
        self,
        model_config: ModelConfig,
        merging_config: MergingConfig,
        datasets: list[str],
        checkpoint_names: list[str],
        device,
    ):
        assert merging_config.fisher_approximation == "diagonal"

        checkpointStatistic_name = []
        for checkpoint_fp in checkpoint_names:
            checkpointStatistic_name.append(
                getCheckpointFisher_name(checkpoint_fp, merging_config)
            )

        loaded_fishers = loadCheckpointOrStatistics_fromNames(
            model_config, datasets, checkpointStatistic_name, device
        )

        return loaded_fishers

    def fisher_merging(self, loaded_checkpoints, loaded_statistics, merging_lambda):
        fisherWeightedCheckpoints_acrossTasks = []
        fishers_acrossTasks = []

        for dataset, checkpoint in loaded_checkpoints.items():
            fisher = loaded_statistics[dataset]

            # If merging 2 checkpoints, allow for a weighted Diagonal Fisher merging
            if len(loaded_checkpoints) == 2:
                if len(fisherWeightedCheckpoints_acrossTasks) == 0:
                    fisher = elementWise_scale(fisher, merging_lambda)
                else:
                    assert len(fisherWeightedCheckpoints_acrossTasks) == 1
                    fisher = elementWise_scale(fisher, (1 - merging_lambda))
            else:
                assert merging_lambda == 1.0

            # Element-wise multiply the Diagonal Fisher with the checkpoint
            fisherWeighted_checkpoint = pairwiseMap_modelParameters(
                checkpoint, fisher, lambda x, y: x * y
            )

            fisherWeightedCheckpoints_acrossTasks.append(fisherWeighted_checkpoint)
            fishers_acrossTasks.append(fisher)

        merged_model = elementWise_divide(
            efficientReduceSum_modelParameters(fisherWeightedCheckpoints_acrossTasks),
            efficientReduceSum_modelParameters(fishers_acrossTasks),
        )

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
        self._checkMergingMethod_inConfig(merging_config)

        # For some weights that are merged (i.e. LoRA), the other weight (i.e. the A matrix) computes the merge and the other weight (i.e. the B matrix) can be ignored
        if len(loaded_parameters) == 0:
            return loaded_parameters, None

        return (
            self.fisher_merging(
                loaded_parameters,
                loaded_parameterStatistics,
                merging_config.merging_lambda,
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
    ) -> Dict[str, Dict]:
        """

        Args:
            loaded_checkpoints:
            loaded_metadata: diagonal Fishers
            merging_lambda:

        Returns:
            merged_checkpoint
        """
        self._checkMergingMethod_inConfig(merging_config)
        assert len(loaded_checkpoints) == len(loaded_statistics)

        return (
            self.fisher_merging(
                loaded_checkpoints, loaded_statistics, merging_config.merging_lambda
            ),
            None,
        )
