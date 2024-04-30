import json
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List

import torch
import wandb

from src.utils.distributed import is_distributedSetup
from src.utils.io import *
from src.utils.utils import *
from src.model.utils import *
from src.model.model_io import *

METRICS_PRIORITY = ["score_to_select_checkpoint", "loss", "batch_idx"]
WANDB_KEYS_TO_IGNORE = {"scores_per_step": True, "predictions_fp": True}


class Checkpointer(object):
    def __init__(self, training_config, initialCheckpoint_idx):
        """
        Args:
            training_config:
            initial_checkpoint:
        """
        self.training_config = training_config
        self.initial_checkpoint = initialCheckpoint_idx

        if training_config.log_wandb:
            exp_name = (
                os.path.dirname(training_config.experiment_dir)
                .replace("exp_out/", "")
                .replace("/", "_")
            )
            os.environ["WANDB_PROGRAM_RELPATH"] = "src/train/training.py"
            # Ignore storing the evaluation dataset key_values since it will
            # override the training dataset key_values since the hyperparameter
            # names are the same
            all_keyValues = training_config.get_key_values()
            all_keyValues.update(training_config.get_datasetConfig().get_key_values())
            all_keyValues.update(training_config.get_modelConfig().get_key_values())
            all_keyValues.update(
                training_config.get_evaluationConfig().get_key_values()
            )

            wandb.init(
                project="mm",
                name=exp_name,
                dir=training_config.experiment_dir,
                resume=True,
                entity="raffel-reports",
                settings=wandb.Settings(
                    disable_git=True,
                    save_code=False,
                ),
                config=all_keyValues,
            )

        self.runningSum_ofMetrics = {}
        self.numberOfUpdates_sinceLastCheckpoint = 0

        self.current_bestScore = 0
        self.numberOfCheckpoints_sinceBestCheckpoint = 0

        self.log_fp = os.path.join(
            self.training_config.experiment_dir, "training_log.json"
        )

        self._get_bestCheckpoint()

    def _get_bestCheckpoint(self):
        """
        If we are resuming training from a checkpoint, we check what the best checkpoint saved so far was
        """
        if os.path.exists(self.log_fp):
            list_scores = read_jsonl(self.log_fp)

            previous_bestScore = 0
            previous_bestCheckpointIdx = 0
            for score in list_scores:
                if score["score_to_select_checkpoint"] > previous_bestScore:
                    previous_bestScore = score["score_to_select_checkpoint"]
                    previous_bestCheckpointIdx = score["batch_idx"]

            # If we are resuming training, the initial checkpoint to resume training
            # should match the last checkpoint sored in the log
            assert list_scores[-1]["batch_idx"] == self.initial_checkpoint

            self.current_bestScore = previous_bestScore
            print(
                self.initial_checkpoint,
                previous_bestCheckpointIdx,
                self.training_config.checkpoint_frequency,
            )

            if previous_bestCheckpointIdx != 0:
                assert (
                    self.initial_checkpoint - previous_bestCheckpointIdx
                ) % self.training_config.checkpoint_frequency == 0

            self.numberOfCheckpoints_sinceBestCheckpoint = (
                self.initial_checkpoint - previous_bestCheckpointIdx
            ) // self.training_config.checkpoint_frequency

            print(self.current_bestScore, self.numberOfCheckpoints_sinceBestCheckpoint)

    def _is_bestCheckpoint(self, current_log: Dict):
        """

        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        return current_score > self.current_bestScore

    def _update_bestCheckpoint(self, current_log: Dict):
        """
        Args:
            current_log:

        Returns:

        """
        current_score = getValueOfKey_inDictionary(current_log, METRICS_PRIORITY)
        self.current_bestScore = current_score
        self.numberOfCheckpoints_sinceBestCheckpoint = 0

    def _save_checkpoint(
        self, trainable_parameters: Dict[str, torch.Tensor], save_name: str
    ):
        """
        Args:
            trainable_parameters:
            save_name:
        """
        save_fp = save_checkpoint(
            trainable_parameters,
            save_name,
        )
        saveTo_gcp(self.training_config.should_save_training_state_to_gcp, save_fp)

    def _save_trainingState(
        self, trainable_parameters, optimizer, scheduler, batch_idx: int, save_fp: str
    ):
        """
        Args:
            trainable_parameters:
            optimizer:
            scheduler:
            batch_idx:
            save_fp:

        Returns:

        """
        current_stateDict = {
            "num_batches": batch_idx,
            "model": trainable_parameters,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }
        torch.save(current_stateDict, save_fp)
        saveTo_gcp(self.training_config.should_save_training_state_to_gcp, save_fp)

    def update_runningSumOfMetrics(self, current_metrics: Dict[str, float]):
        """
        Args:
            current_metrics:
        """
        self.runningSum_ofMetrics = addValues_inDict(
            self.runningSum_ofMetrics, current_metrics
        )
        self.numberOfUpdates_sinceLastCheckpoint += 1

    def _get_averageMetrics(self):
        """
        Get average metric per batch since the last time we got the average.

        Note that average is per example, not batch size
        (i.e. every gradient update, not every forward pass).
        """
        average_metric = {}
        for k in self.runningSum_ofMetrics.keys():
            average_metric[k] = float(
                "%.3f"
                % (
                    self.runningSum_ofMetrics[k]
                    / self.numberOfUpdates_sinceLastCheckpoint
                    / self.training_config.train_batch_size
                )
            )

        # Reset running dict_metrics and counter when we take average
        self.runningSum_ofMetrics = {}
        self.numberOfUpdates_sinceLastCheckpoint = 0

        return average_metric

    def _filterLog_forWandb(self, current_log: Dict) -> Dict:
        """
        Args:
            current_log:

        Returns:

        """
        wandb_log = {}
        for key, value in current_log.items():
            if key not in WANDB_KEYS_TO_IGNORE:
                wandb_log[key] = value
        return wandb_log

    def _log_metricAndScores(self, batch_idx: int, evaluation_scores: Dict) -> Dict:
        """
        Args:
            batch_idx:
            evaluation_scores:

        Returns:
            _description_
        """
        current_log = {}
        current_log["batch_idx"] = batch_idx
        current_log.update(self._get_averageMetrics())
        current_log.update(evaluation_scores)

        append_json(current_log, self.log_fp, pretty_print=False)

        if self.training_config.log_wandb:
            wandb.log(self._filterLog_forWandb(current_log))

        saveTo_gcp(self.training_config.should_save_training_state_to_gcp, self.log_fp)

        return current_log

    def checkpoint(
        self,
        trainable_parameters: Dict[str, torch.Tensor],
        optimizer,
        scheduler,
        evaluation_scores: Dict,
        batch_idx: int,
    ):
        """
        Handles checkpointing which means
        1) logging metrics and evaluation_scores
        2) saving the model if needed

        Args:
            trainable_parameters:
            optimizer,
            scheduler,
            evaluation_scores:
            batch_idx:

        Returns:
            current_log
        """
        current_log = self._log_metricAndScores(batch_idx, evaluation_scores)

        self.numberOfCheckpoints_sinceBestCheckpoint += 1

        # Save training state in training state directory which is symlinked to other location with more memory
        trainingState_dir = os.path.join(
            self.training_config.experiment_dir, "training_state"
        )
        if not os.path.exists(trainingState_dir):
            if self.training_config.symlink_training_state_directory:
                assert self.training_config.slurm_job_id is not None
                symlink_directory = os.path.join(
                    f"/checkpoint/dertam/{self.training_config.slurm_job_id}",
                    f"{trainingState_dir}",
                )
                if not os.path.exists(symlink_directory):
                    os.makedirs(symlink_directory)
                    os.symlink(symlink_directory, trainingState_dir)
            else:
                os.makedirs(trainingState_dir)

        # Create checkpoint directory (and symlink if needed)
        checkpoint_dir = os.path.join(
            self.training_config.experiment_dir, "checkpoints"
        )
        if not os.path.exists(checkpoint_dir):
            if self.training_config.symlink_checkpoint_state_directory:
                assert self.training_config.slurm_job_id is not None
                symlink_directory = os.path.join(
                    f"/checkpoint/dertam/{self.training_config.slurm_job_id}",
                    f"{checkpoint_dir}",
                )
                if not os.path.exists(symlink_directory):
                    os.makedirs(symlink_directory)
                    os.symlink(symlink_directory, checkpoint_dir)
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)

        # Save every checkpoint
        if self.training_config.should_save_every_checkpoint:
            self._save_checkpoint(
                trainable_parameters,
                os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}"),
            )

        # Save only best checkpoint
        if self._is_bestCheckpoint(current_log):
            deleteFiles_inDirectory(checkpoint_dir, "best")
            self._save_checkpoint(
                trainable_parameters,
                os.path.join(checkpoint_dir, f"best_checkpoint_{batch_idx}"),
            )

        # Ignore saving the model if we are just evaluating at the beginning
        if batch_idx > 0:
            if self.training_config.should_save_training_state:
                deleteFiles_inDirectory(trainingState_dir, "training_state")
                self._save_trainingState(
                    trainable_parameters,
                    optimizer,
                    scheduler,
                    batch_idx,
                    os.path.join(trainingState_dir, f"training_state_{batch_idx}.pt"),
                )

        if self._is_bestCheckpoint(current_log):
            self._update_bestCheckpoint(current_log)

        logging.info(f"Finished {batch_idx} batches with log {current_log}")

        return self.numberOfCheckpoints_sinceBestCheckpoint
