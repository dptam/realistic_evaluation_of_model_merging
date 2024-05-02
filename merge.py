import argparse
import copy
import logging
import os
import time
from tqdm import tqdm

import torch

from src.utils.io import *
from src.eval.inference import inference
from src.merging.utils.model_ops import *
from src.merging.utils.utils import *
from src.merging.utils.model_utils import *
from src.model.ModelConfig import ModelConfig
from src.model.utils import *
from src.train.TrainingConfig import TrainingConfig
from src.utils.config_utils import *

from src.merging.methods.Average import Average
from src.merging.methods.TaskArithmetic import TaskArithmetic
from src.merging.methods.DARETaskArithmetic import DARETaskArithmetic
from src.merging.methods.FisherMerging import FisherMerging
from src.merging.methods.RegMean import RegMean
from src.merging.methods.MaTS import MaTS
from src.merging.methods.TIES import TIES

MERGING_METHODS = {
    "average": Average,
    "task_arithmetic": TaskArithmetic,
    "dare_task_arithmetic": DARETaskArithmetic,
    "fisher_merging": FisherMerging,
    "regmean": RegMean,
    "mats": MaTS,
    "ties": TIES,
}


def get_mergingMethod(merging_config):
    return MERGING_METHODS[merging_config.method]()


def synchronous_merge(
    merging_method,
    model_config,
    merging_config,
    loaded_checkpoints,
    loaded_checkpointStatistics,
    mergedCheckpoint_name,
    device,
):
    merged_model, merged_log = merging_method.synchronous_merge(
        model_config,
        merging_config,
        loaded_checkpoints,
        loaded_checkpointStatistics,
    )
    if merging_config.save_model:
        save_checkpoint(merged_model, mergedCheckpoint_name, store_asynchronously=False)

    if merged_log is not None:
        with open(
            os.path.join(mergedCheckpoint_name, "log.json"),
            "w+",
        ) as f:
            f.write(json.dumps(merged_log, indent=4) + "\n")

    return merged_model


def asynchronous_merge(
    merging_method,
    model_config,
    merging_config,
    datasets,
    checkpoint_metadatas,
    parameter_names,
    mergedCheckpoint_name,
    device,
):
    merged_model = {}
    merged_log = None

    parameters_toIgnore = getParameterNames_toIgnore(model_config)

    final_parameterNames = getFinal_parameterNames(
        model_config, merging_config, list(parameter_names)
    )

    all_parameterNames = []

    # new_finalParameterNames = {}
    # new_finalParameterNames["transformer.decoder.block.0.layer.0.SelfAttention.k.weight"] = final_parameterNames["transformer.decoder.block.0.layer.0.SelfAttention.k.weight"]
    # final_parameterNames = new_finalParameterNames
    # print(final_parameterNames)
    for idx, (final_parameterName, original_parameterNames) in tqdm(
        enumerate(final_parameterNames.items())
    ):
        # Skip parameter if needed
        ignore_parameter = False
        for parameter_toIgnore in parameters_toIgnore:
            if re.fullmatch(parameter_toIgnore, final_parameterName):
                ignore_parameter = True

        if ignore_parameter:
            print(f"Ignoring {final_parameterName}")
            continue

        loaded_parameters = loadParameterOrStatistic_fromMetadatas(
            model_config,
            merging_config,
            datasets,
            checkpoint_metadatas,
            final_parameterName,
            original_parameterNames,
            "parameter",
            device,
        )

        if merging_method.requires_asynchronousStatistic():
            loaded_parameterStatistics = merging_method.load_parameterStatistic(
                model_config,
                merging_config,
                datasets,
                checkpoint_metadatas,
                final_parameterName,
                original_parameterNames,
                device,
            )
        else:
            loaded_parameterStatistics = None

        with torch.no_grad():
            merged_parameter, mergedParameter_log = merging_method.asynchronous_merge(
                model_config,
                merging_config,
                loaded_parameters,
                loaded_parameterStatistics,
                final_parameterName,
                original_parameterNames,
                device,
            )
        torch.cuda.empty_cache()
        merged_model.update(merged_parameter)
        if mergedParameter_log is not None:
            if merged_log is None:
                merged_log = {}
            merged_log.update(mergedParameter_log)

        if merging_config.asynchronous_frequency_to_save_parameters is not None and (
            idx % merging_config.asynchronous_frequency_to_save_parameters == 0
            or re.search("transformer\.(lm_head|shared)\.weight", final_parameterName)
        ):
            _ = save_checkpoint(
                merged_model,
                mergedCheckpoint_name,
                model_config.store_asynchronously,
                None,
            )
            all_parameterNames.extend(list(merged_model.keys()))
            del merged_model
            merged_model = {}

    all_parameterNames.extend(list(merged_model.keys()))
    if merging_config.save_model:
        save_checkpoint(
            merged_model,
            mergedCheckpoint_name,
            model_config.store_asynchronously,
            all_parameterNames,
        )
    if merged_log is not None:
        with open(
            os.path.join(os.path.dirname(mergedCheckpoint_name), "log.json"),
            "w+",
        ) as f:
            f.write(json.dumps(merged_log, indent=4) + "\n")

    return merged_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = addConfigArguments_toParser(
        parser,
        add_trainingArguments=False,
        add_inferenceArguments=True,
        add_mergingArguments=True,
    )
    parser.add_argument("--additional_task_mixture_evaluation", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_config, evaluation_config, merging_config = construct_configs(
        args, "eval", is_merging=True
    )

    assert merging_config.task_mixture is not None

    # Load config that will merge the weights
    if merging_config.merge_peft_weights:
        model_config = getNewModelConfig_withMergedWeights(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cached_singleDatasetReaders = {}
    cached_models = {}
    # Loop through mixture subsets
    for outerIterate_mergingConfig in getMergingConfig_withDifferentMixtureSubsetIds(
        merging_config
    ):
        print(outerIterate_mergingConfig.get_key_values())

        datasets, checkpoint_names = getCheckpointNames_inTaskMixture(
            model_config,
            outerIterate_mergingConfig,
        )

        merging_method = get_mergingMethod(outerIterate_mergingConfig)
        # asynchronous merging
        if outerIterate_mergingConfig.merge_asynchronously:
            model_config = update_modelConfig(
                model_config, {"store_asynchronously": True}
            )
            (
                checkpoint_metadatas,
                parameter_names,
            ) = loadCheckpointMetadatas_fromNames(
                model_config,
                checkpoint_names,
            )
        # synchronous merging
        else:
            assert outerIterate_mergingConfig.merge_asynchronously == False

            loaded_checkpoints = loadCheckpointOrStatistics_fromNames(
                model_config,
                datasets,
                checkpoint_names,
                device,
            )

            if merging_method.requires_synchronousStatistic():
                loaded_checkpointStatistics = merging_method.load_parameterStatistic(
                    model_config,
                    outerIterate_mergingConfig,
                    datasets,
                    checkpoint_names,
                    device,
                )
            else:
                loaded_checkpointStatistics = None

        for (
            innerIterate_mergingConfig
        ) in merging_method.getMergingConfigs_withDifferentHyperparameters(
            outerIterate_mergingConfig
        ):
            print(innerIterate_mergingConfig.get_key_values())

            new_evaluationConfig = update_evaluationConfig(
                evaluation_config,
                {},
                {
                    "task_mixture": innerIterate_mergingConfig.task_mixture,
                    "mixture_subset_size": innerIterate_mergingConfig.mixture_subset_size,
                    "mixture_subset_id": innerIterate_mergingConfig.mixture_subset_id,
                },
            )
            # Construct merging checkpoint dir
            experiment_dir = getMerging_experimentDir(
                model_config,
                new_evaluationConfig.get_datasetConfig().instruction_format,
                innerIterate_mergingConfig,
            )
            os.makedirs(experiment_dir, exist_ok=True)

            innerIterate_mergingConfig._save_config(
                os.path.join(experiment_dir, "merging_config.json"),
                shouldSave_toGCP=False,
            )

            mergedCheckpoint_name = os.path.join(experiment_dir, "merged_model")
            # Asynchronous merge of each parameter invidiually
            if merging_config.merge_asynchronously:
                merged_model = asynchronous_merge(
                    merging_method,
                    model_config,
                    innerIterate_mergingConfig,
                    datasets,
                    checkpoint_metadatas,
                    parameter_names,
                    mergedCheckpoint_name,
                    device,
                )
                model_updateDict = {"filepath_to_load_model": mergedCheckpoint_name}

                if merging_config.asynchronous_frequency_to_save_parameters is None:
                    cached_models[mergedCheckpoint_name] = merged_model

            # Synchronous merge doesn't need to pass in merge_peft_weights since loading the checkpoints will already handle merging weights
            else:
                merged_model = synchronous_merge(
                    merging_method,
                    model_config,
                    innerIterate_mergingConfig,
                    loaded_checkpoints,
                    loaded_checkpointStatistics,
                    mergedCheckpoint_name,
                    device,
                )
                model_updateDict = {"filepath_to_load_model": mergedCheckpoint_name}
                cached_models[mergedCheckpoint_name] = merged_model
            # Load model config for evaluation
            new_modelConfig = getNewModelConfig_withLoadedWeights(
                model_config, merging_config.merge_peft_weights, model_updateDict
            )

            if new_evaluationConfig.mixture_subset_id is not None:
                title = f"Mixture Subset {new_evaluationConfig.mixture_subset_id}"
            else:
                title = None

            # Doing inference on model
            cached_models, cached_singleDatasetReaders = inference(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                1,
                None,
                new_modelConfig,
                new_evaluationConfig,
                experiment_dir,
                title=title,
                cached_models=cached_models,
                cached_singleDatasetReaders=cached_singleDatasetReaders,
            )

            if args.additional_task_mixture_evaluation:
                additional_evaluationConfig = update_evaluationConfig(
                    evaluation_config,
                    {},
                    {"task_mixture": args.additional_task_mixture_evaluation},
                )

                cached_models, cached_singleDatasetReaders = inference(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    1,
                    None,
                    new_modelConfig,
                    additional_evaluationConfig,
                    experiment_dir,
                    title=title,
                    cached_models=cached_models,
                    cached_singleDatasetReaders=cached_singleDatasetReaders,
                )

            del cached_models[mergedCheckpoint_name]
