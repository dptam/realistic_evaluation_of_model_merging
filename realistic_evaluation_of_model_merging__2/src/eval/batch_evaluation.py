import logging
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from tqdm import tqdm

from src.data.dataset_reader.task_mixtures import *
from src.utils.config_utils import *
from src.eval.EvaluationConfig import EvaluationConfig
from src.data.batches import getSingleEpoch_OfBatches
from src.data.readers import get_datasetReader
from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.DatasetConfig import DatasetConfig
from src.data.Dataset import LanguageDataset, VisionDataset
from src.eval.Evaluator import Evaluator
from src.eval.utils import *
from src.utils.distributed import (
    is_distributedSetup,
    is_nodeZero,
    reduce_gatheredOutput,
)


def batchEvaluate_fromConfig(
    batchOf_models,
    batchOf_modelConfigs: ModelConfig,
    evaluation_config: EvaluationConfig,
    batchOf_predictionDir: str,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
    world_size: int,
    device,
) -> (Dict, Dict[str, DatasetReader]):
    """

    Args:
        model:
        evaluation_config:
        batchOf_predictionDir:
        cached_singleDatasetReaders:
        world_size
        device:

    Returns:

    """
    logging.info(f"Evaluating model")

    evaluationDataset_config = evaluation_config.get_datasetConfig()
    dataset_reader, cached_singleDatasetReaders = get_datasetReader(
        task_mixture=None,
        mixtureSubset_size=None,
        mixtureSubset_id=None,
        dataset_config=evaluationDataset_config,
        cached_singleDatasetReaders=cached_singleDatasetReaders,
    )

    for model in batchOf_models:
        model.eval()

    # Ignoring distributed setup for batch (of models) evaluation
    assert not is_distributedSetup(world_size)

    language_or_vision = None
    for model_config in batchOf_modelConfigs:
        if language_or_vision is None:
            language_or_vision = model_config.language_or_vision
        else:
            assert language_or_vision == model_config.language_or_vision

        if model_config.language_or_vision == "language":
            dataset = LanguageDataset(
                dataset_reader.get_dataset("eval"),
                evaluationDataset_config,
                batchOf_models[0].tokenize_fn,
                "eval",
                device=device,
            )
        else:
            dataset = VisionDataset(
                dataset_reader.get_dataset("eval"),
                evaluationDataset_config,
                batchOf_models[0].preprocess_fn,
                "eval",
                device=device,
            )

    evalBatch_iterator = getSingleEpoch_OfBatches(
        dataset, evaluation_config.eval_batch_size
    )
    metrics = dataset_reader.get_datasetMetrics()

    if is_nodeZero(device):
        batchOf_evaluators = []
        for model_idx in range(len(batchOf_models)):
            evaluator = Evaluator(
                evaluation_config,
                metrics,
                os.path.join(
                    batchOf_predictionDir[model_idx],
                    evaluation_config.get_datasetConfig().split,
                ),
            )
            batchOf_evaluators.append(evaluator)

    with torch.no_grad():
        for batch in tqdm(evalBatch_iterator):
            batchOf_evalInfo = prepare_batchOfEvalInfo(batch)

            for model_idx, model in enumerate(batchOf_models):
                copyOfBatchOf_evalInfo = copy.deepcopy(batchOf_evalInfo)
                if "Accuracy" in metrics:
                    if language_or_vision == "language":
                        (
                            predicted_choice,
                            score_ofChoices,
                            logProbs_ofAllChoicesIds,
                            len_allChoices,
                        ) = model.predict_mulChoice(
                            batch, evaluation_config.length_normalization
                        )

                        copyOfBatchOf_evalInfo.update(
                            {
                                "predicted_choice": predicted_choice,
                                "score_of_choices": score_ofChoices,
                                "log_probs_of_all_choices_ids": logProbs_ofAllChoicesIds,
                                "len_all_choices": len_allChoices,
                            }
                        )
                    else:
                        assert language_or_vision == "vision"
                        (
                            predicted_choice,
                            predicted_logProb,
                        ) = model.predict(batch)

                        copyOfBatchOf_evalInfo.update(
                            {
                                "predicted_choice": predicted_choice,
                                "predicted_log_prob": predicted_logProb,
                            }
                        )

                if (
                    "Squad" in metrics
                    or "F1" in metrics
                    or "arithmetic" in metrics
                    or "kv_substitution" in metrics
                    or "kv_substitution_arithmetic" in metrics
                    or "sp_rouge" in metrics
                ):
                    generated_ids, generated_txt = model.generate(
                        batch,
                        evaluation_config.max_gen_len,
                        evaluation_config.sample_tokens,
                    )

                    copyOfBatchOf_evalInfo.update(
                        {
                            "generated_ids": generated_ids,
                            "prediction_text": generated_txt,
                        }
                    )

                    # Append some additional input to the generation and ask the model to continue generating
                    # This is needed for KVArithmetic in particular.
                    if "additional_input" in batch:
                        copyOfBatchOf_evalInfo = additionalRound_ofGeneration(
                            batchOf_models,
                            evaluation_config,
                            batch,
                            batchOf_evalInfo,
                            device,
                        )

                if is_nodeZero(device):
                    batchOf_evaluators[model_idx].add_batch(copyOfBatchOf_evalInfo)

    if is_nodeZero(device):
        batchOf_results = []
        for evaluator in batchOf_evaluators:
            results = {
                "score": {
                    evaluation_config.get_datasetConfig().split: evaluator.get_result(),
                },
                "evaluation_dir": evaluator.get_evaluationRunDir(),
                "evaluation_config": evaluation_config.get_key_values(),
                "dataset_config": evaluation_config.get_datasetConfig().get_key_values(),
            }
            batchOf_results.append(results)
        return (
            batchOf_results,
            cached_singleDatasetReaders,
        )
    else:
        return None, cached_singleDatasetReaders


def batchEvaluate_onDatasets(
    batchOf_models,
    batchOf_modelConfigs: ModelConfig,
    evaluation_config: EvaluationConfig,
    batchOf_predictionDir: str,
    cached_singleDatasetReaders: Dict[str, DatasetReader],
    world_size: int,
    device,
) -> (Dict, List[str], Dict[str, DatasetReader]):
    """

    Args:
        batchOf_models:
        batchOf_modelConfigs:
        evaluation_config:
        prediction_dir:
        cached_singleDatasetReaders:
        world_size:
        device:

    Returns:
        score:
        runs_dir:
        cached_singleDatasetReaders:
    """
    score = None
    evaluation_dir = None

    # Evaluate each dataset in the mixture separately
    if evaluation_config.task_mixture is not None:
        all_results = []

        for dataset in getTasks_inMixture(
            evaluation_config.task_mixture,
            evaluation_config.mixture_subset_size,
            evaluation_config.mixture_subset_id,
        ):
            dataset_evaluationConfig = update_evaluationConfig(
                evaluation_config, getDatasetUpdateDict_fromTask(dataset), {}
            )
            batchOf_results, cached_singleDatasetReaders = batchEvaluate_fromConfig(
                batchOf_models,
                batchOf_modelConfigs,
                dataset_evaluationConfig,
                batchOf_predictionDir,
                cached_singleDatasetReaders,
                world_size,
                device,
            )
            all_results.append(batchOf_results)

        # Check result is not None since for DDP, result will be None except for the node 0
        if all_results[0] is not None:

            # all results is list of dataset results where each dataset result is a list of model result. transposed results is a list of model results where each model result is a list of dataset result
            transposed_allResults = list(map(list, zip(*all_results)))

            batchOf_scores = []
            batchOf_evaluationDirs = []

            for model_result in transposed_allResults:

                average_score = average_scores(
                    model_result, evaluation_config.get_datasetConfig().split
                )

                def getDataset_fn(dataset_score):
                    dataset_name = dataset_score["dataset_config"]["dataset"]
                    # Account for cross-lingual dataset where we add the language code to the task to construct dataset name to look up checkpoint
                    language_code = dataset_score["dataset_config"]["language_code"]
                    if language_code is not None:
                        return dataset_name + "-" + language_code
                    language = dataset_score["dataset_config"]["language"]
                    if language is not None:
                        return dataset_name + "-" + language
                    # Account for different domains dataset where we add the domain to the task to construct dataset name to look up checkpoint
                    domain = dataset_score["dataset_config"]["domain"]
                    if domain is not None:
                        task = dataset_score["dataset_config"]["task"]
                        return dataset_name + "-" + domain + "-" + str(task)

                    return dataset_name

                score = concatenate_scores(model_result, getDataset_fn)
                score = deep_update(score, average_score)
                evaluation_dir = get_allRunsDirs(model_result)
                batchOf_scores.append(score)
                batchOf_evaluationDirs.append(evaluation_dir)

    # Evaluate single dataset
    else:
        assert evaluation_config.get_datasetConfig().dataset is not None

        batchOf_results, cached_singleDatasetReaders = batchEvaluate_fromConfig(
            batchOf_models,
            batchOf_modelConfigs,
            evaluation_config,
            batchOf_predictionDir,
            cached_singleDatasetReaders,
            world_size,
            device,
        )
        # Check result is not None since for DDP, result will be None except for the node 0
        if batchOf_results is not None:

            batchOf_scores = []
            batchOf_evaluationDirs = []

            for result in batchOf_results:
                score = result["score"]
                evaluation_dir = result["evaluation_dir"]
                batchOf_scores.append(score)
                batchOf_evaluationDirs.append(evaluation_dir)

    return batchOf_scores, batchOf_evaluationDirs, cached_singleDatasetReaders
