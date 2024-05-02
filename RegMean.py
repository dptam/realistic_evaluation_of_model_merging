import argparse
import copy
import logging
import os

import torch

from src.eval.inference import *
from src.merging.save_statistic import *
from src.merging.methods.MergingMethod import MergingMethod
from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.model.utils import *


class RegMean(MergingMethod):
    def __init__(
        self,
    ):
        super().__init__()

    def _checkMergingMethod_inConfig(self, merging_config):
        assert merging_config.method == "regmean"

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
        finalParameter_name: list[str],
        original_parameterNames: list[str],
        device,
    ):
        assert merging_config.fisher_approximation == "input_activations_covariance"
        if not self._isGramMatrix_computedOnParameter(finalParameter_name):
            return {}

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
            self._getGramMatrix_parameterName(finalParameter_name),
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
        assert merging_config.fisher_approximation == "input_activations_covariance"

        checkpointStatistic_name = []
        for checkpoint_fp in checkpoint_names:
            checkpointStatistic_name.append(
                getCheckpointFisher_name(checkpoint_fp, merging_config)
            )

        loaded_gramMatrix = loadCheckpointOrStatistics_fromNames(
            model_config, datasets, checkpointStatistic_name, device
        )

        return loaded_gramMatrix

    def _getGramMatrix_parameterName(self, parameter_name):
        """
        Map parameter name to the name used to store the corresponding Gram matrix

        Args:
            parameter_name:

        Returns:

        """
        # CLIP Multihead attention has a different weight format
        if re.search("clip.*attn.in_proj_weight", parameter_name):
            fisher_parameterName = parameter_name.replace(".in_proj_weight", "")
        else:
            fisher_parameterName = parameter_name.replace(".weight", "")
        return fisher_parameterName

    def _isGramMatrix_computedOnParameter(self, parameter_name):
        """
        Map parameter name to the name used to store the corresponding Gram matrix

        Args:
            parameter_name:

        Returns:

        """

        if re.search(
            ".*(?:bias|ln_1|ln_2|visual.proj|positional_embedding|class_embedding|ln_post|visual.conv1|ln_pre|layer_norm|transformer.shared).*",
            parameter_name,
        ):
            return False
        else:
            return True

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

        if len(loaded_parameterStatistics) == 0:
            print(f"Averaging {finalParameter_name}")
            sum_parameters = efficientReduceSum_modelParameters(
                list(loaded_parameters.values())
            )
            merged_parameter = elementWise_scale(
                sum_parameters, 1 / len(loaded_parameters)
            )
        else:
            gramMatrixTimesWeights_acrossTasks = []
            gramMatrix_acrossTasks = []

            for dataset, parameter in loaded_parameters.items():
                parameter_value = parameter[finalParameter_name]
                gram_matrix = loaded_parameterStatistics[dataset][
                    self._getGramMatrix_parameterName(finalParameter_name)
                ]

                scaled_gramMatrix = scale_nonDiagonalElements(
                    gram_matrix, merging_config.merging_lambda
                )
                if scaled_gramMatrix.dtype != parameter_value.dtype:
                    parameter_value = parameter_value.to(scaled_gramMatrix.dtype)

                scaledGram_timesWeight = torch.matmul(
                    scaled_gramMatrix, parameter_value.T
                )

                gramMatrix_acrossTasks.append({finalParameter_name: scaled_gramMatrix})
                gramMatrixTimesWeights_acrossTasks.append(
                    {finalParameter_name: scaledGram_timesWeight}
                )

            # Compute the final weights according to RegMean
            merged_parameter = matrix_multiply(
                matrix_inverse(
                    efficientReduceSum_modelParameters(gramMatrix_acrossTasks)
                ),
                efficientReduceSum_modelParameters(gramMatrixTimesWeights_acrossTasks),
            )

            for finalParameter_name, parameter in merged_parameter.items():
                merged_parameter[finalParameter_name] = parameter.T
  

        return merged_parameter, None

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
            model_lambda:
            checkpoint_andFisherMatrices:
        Returns:

        """
        self._checkMergingMethod_inConfig(merging_config)
        gramMatrixTimesWeights_acrossTasks = []
        gramMatrix_acrossTasks = []
        nonGramMatrixWeights_acrossTasks = []

        for (
            dataset,
            checkpoint,
        ) in loaded_checkpoints.items():
            gram_matrices = {}
            gramMatrix_timesWeight = {}
            nonGramMatrix_weights = {}
            for parameter_name, parameter in checkpoint.items():
                regMean_parameterName = self._getGramMatrix_parameterName(
                    parameter_name
                )

                # Store scaled gram matrix and gram matrix times weight
                if regMean_parameterName in loaded_statistics[dataset]:
                    gram_matrix = loaded_statistics[dataset][regMean_parameterName]
                    scaled_gramMatrix = scale_nonDiagonalElements(
                        gram_matrix, merging_config.merging_lambda
                    )

                    gram_matrices[parameter_name] = scaled_gramMatrix

                    # Transpose the weight since Pytorch stores the transposed weight by default
                    if scaled_gramMatrix.dtype != parameter.dtype:
                        parameter = parameter.to(scaled_gramMatrix.dtype)

                    scaledGram_timesWeight = torch.matmul(
                        scaled_gramMatrix, parameter.T
                    )
                    gramMatrix_timesWeight[parameter_name] = scaledGram_timesWeight
                # Default to simple averaging
                else:
                    nonGramMatrix_weights[parameter_name] = parameter

            gramMatrix_acrossTasks.append(gram_matrices)
            gramMatrixTimesWeights_acrossTasks.append(gramMatrix_timesWeight)
            nonGramMatrixWeights_acrossTasks.append(nonGramMatrix_weights)

        # Compute the final weight by inverting sum of gram matrices and multiplying by gram matrix
        # multiplied by weight
        merged_model = matrix_multiply(
            matrix_inverse(efficientReduceSum_modelParameters(gramMatrix_acrossTasks)),
            efficientReduceSum_modelParameters(gramMatrixTimesWeights_acrossTasks),
        )
        sum_nonGramMatrixWeights = efficientReduceSum_modelParameters(
            nonGramMatrixWeights_acrossTasks
        )
        merged_nonGramMatrixWeights = elementWise_scale(
            sum_nonGramMatrixWeights, 1 / len(nonGramMatrixWeights_acrossTasks)
        )
        # Transpose back the parameters since they were transposed earlier
        for parameter_name, parameter in merged_model.items():
            merged_model[parameter_name] = parameter.T

        # Add the non-merged weights
        for parameter_name, parameter in merged_nonGramMatrixWeights.items():
            merged_model[parameter_name] = parameter

        return merged_model, None
