from typing import Any, Callable, Dict, List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ModelConfig import ModelConfig
from src.model.utils import *
from src.model.utils import computeLogProb_perChoice
from src.utils.stats import *
from src.utils.utils import *


class DecoderWrapper(nn.Module):
    def __init__(self, transformer, tokenizer, model_config: ModelConfig):
        """

        Args:
            transformer:
            tokenizer:
            model_config:
        """
        super().__init__()
        self.transformer = transformer
        self.leftPadding_tokenizer = tokenizer
        self.input_tokenizer = tokenizer
        self.decoderInput_tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer = self.decoderInput_tokenizer
        self.target_tokenizer = copy.deepcopy(tokenizer)

        # Use eos_token for pad_token if it doesn't exist. This is ok since the
        # pad tokens will be ignored through the mask
        if self.input_tokenizer.pad_token_id is None:
            self.input_tokenizer.pad_token_id = self.input_tokenizer.eos_token_id
        if self.decoderInput_tokenizer.pad_token_id is None:
            self.decoderInput_tokenizer.pad_token_id = (
                self.decoderInput_tokenizer.eos_token_id
            )
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id

        self.model_config = model_config

        self.input_tokenizer.padding_side = "left"

        self.decoderInput_tokenizer.padding_side = "left"
        self.decoderInput_tokenizer.add_eos_token = True

        self.target_tokenizer.padding_side = "right"
        self.target_tokenizer.add_bos_token = False
        self.target_tokenizer.add_eos_token = True

    def _broadcast_tensors(
        self, input_masks: torch.Tensor, past_key_values: tuple, num_choices: int
    ) -> (torch.Tensor, tuple):
        """
        Broadcast the input masks and encoder outputs to account for multiple choices per input

        Args:
            input_masks: [batch_size, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size, max_input_len, num_heads, head_dim] or [batch_size x num_heads, head_dim, max_input_len].
            num_choices:

        Returns:
            input_masks: [batch_size x num_choices, max_input_len]
            past_key_values: Tuple of keys and values for each layer.
                The first index of the tuple is the layer index, and the second index
                of the tuple is whether it is a key or value. Each element in tuple
                has shape [batch_size x num_choices, max_input_len, num_heads, head_dim]
                or [batch_size x num_heads x num_choices, head_dim, max_input_len].
        """
        batch_size, max_input_len = input_masks.shape
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)

        list_broadcast_pastKeyValues = []
        for pastKeyValues_perLayer in past_key_values:
            list_broadcast_pastKeyValues_perLayer = []
            for key_or_value in pastKeyValues_perLayer:
                # This is for keys or values which have dimension [batch_size, max_input_len, num_heads, head_dim]
                # This is the standard for Hugging Face.
                if len(key_or_value.shape) == 4:
                    list_broadcast_pastKeyValues_perLayer.append(
                        torch.repeat_interleave(key_or_value, num_choices, dim=0)
                    )
                # This is for keys or values which have dimension [batch_size x num_heads, head_dim, max_input_len].
                # This is what is used for BLOOM in transformers == 4.22.0
                elif len(key_or_value.shape) == 3:
                    num_heads = key_or_value.shape[0] // batch_size
                    flatten_keyOrValue = key_or_value.reshape(
                        ((batch_size, num_heads) + key_or_value.shape[1:])
                    )
                    broadcast_flatten_keyOrValue = torch.repeat_interleave(
                        flatten_keyOrValue, num_choices, dim=0
                    )
                    list_broadcast_pastKeyValues_perLayer.append(
                        broadcast_flatten_keyOrValue.flatten(0, 1)
                    )
                else:
                    raise ValueError(
                        f"Invalid cached key or value shape: ", key_or_value.shape
                    )

            list_broadcast_pastKeyValues.append(
                tuple(list_broadcast_pastKeyValues_perLayer)
            )

        return input_masks, tuple(list_broadcast_pastKeyValues)

    def _casualLM_loss(
        self,
        decoder_inputIds: torch.Tensor,
        decoder_inputMask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute casual LM loss for decoder

        Args:
            decoder_inputIds: [batch_size, input_len + target_len]
            decoder_inputMask: [batch_size, input_len + target_len]
            target_mask: [batch_size, target_len]

        Returns:

        """
        outputs = self.transformer(
            input_ids=decoder_inputIds,
            attention_mask=decoder_inputMask,
        )
        # [batch_size, input_len + target_len, vocab_size]
        decoderInput_logits = outputs.logits

        target_len = target_mask.shape[-1]
        input_len = decoder_inputMask.shape[-1] - target_len

        # If the sequence has to be truncated and only the target is kept while the input is
        # truncated away, then the input_len will be 0. In this case, we change input_len to 1
        # and the model is trained to predict the whole target
        if input_len == 0:
            input_len = 1

        # [batch_size, target_len, vocab_size]
        target_logits = decoderInput_logits[:, input_len - 1 : -1, :]

        # The target_ids are not target_ids in batch, but the ones in
        # decoder_input_ids since target_ids will have a <bos> token at the
        # beginning of the target.
        # The target_ids will be used to compute the number of tokens for the
        # loss since the <bos> token at the beginning accounts for the <eos>
        # token at the end of decoder_input_ids

        # [batch_size, target_len, vocab_size]
        target_ids = decoder_inputIds[
            :,
            input_len:,
        ]

        # Compute the log probability for all the decoder_input_ids except the <eos> token
        # [batch_size, target_len, ]
        logProbs_forTargetIds = F.cross_entropy(
            target_logits.flatten(0, 1),
            target_ids.flatten(0, 1),
            reduction="none",
        )

        flattened_targetMask = target_mask.flatten(0, 1)
        loss = torch.sum(logProbs_forTargetIds * flattened_targetMask) / torch.sum(
            flattened_targetMask
        )
        return loss

    def get_inputTokenizer(self):
        return self.input_tokenizer

    def forward(self, batch: Dict) -> (torch.Tensor, Dict):
        loss = self._casualLM_loss(
            batch["decoder_input_ids"],
            batch["decoder_input_mask"],
            batch["target_mask"],
        )

        current_metrics = {"loss": loss.detach().cpu().item()}
        return loss, current_metrics

    def predict_mulChoice(
        self, batch: Dict[str, Any], length_normalization: bool
    ) -> (list[float], list[list[float]], list[list[list[float]]], list[float]):
        """

        Args:
            batch:
            lengthNormalization:

        Returns:
            pred_choice: [batch_size, ]
            score_ofChoices: [batch_size, num_choices]
            logProbs_ofAllChoicesIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size]
        """

        output = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            use_cache=True,
        )
        past_key_values = output.past_key_values

        num_ofAnswerChoices = (
            batch["flattened_answer_choices_ids"].shape[0]
            // batch["input_mask"].shape[0]
        )
        input_masks, past_key_values = self._broadcast_tensors(
            batch["input_mask"], past_key_values, num_ofAnswerChoices
        )

        # Combine the input mask and choice mask so the model knows which cached input representations
        # are padded when conditioning on the cached input representations.
        # [batch_size x num_choices, max_input_len + max_choice_len]
        combined_mask = torch.cat(
            [input_masks, batch["flattened_answer_choices_mask"]], dim=1
        )

        # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
        # pad token of 0 and so the loss will not be ignored for the pad tokens
        transformer_outputs = self.transformer(
            input_ids=batch["flattened_answer_choices_ids"],
            attention_mask=combined_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # We used the logits for all choices to compute the log probs per example since
        # the loss returned in transformer_outputs will average the negative log probs across
        # examples
        # [batch_size x num_choices, max_choice_len, vocab_size]
        logits_ofAnswerChoicesIds = transformer_outputs.logits.float()
        vocab_size = logits_ofAnswerChoicesIds.shape[-1]

        # Shift the ids, masks, logits to handle predicting the next token for the decoder.
        # Note that we need to pass in the input_ids and cannot rely on HuggingFace automatically
        #   constructing the ids from the labels, since we need to pass in an attention mask to handle
        #   the cached input representations.
        shiftedLogits_ofAllChoices = logits_ofAnswerChoicesIds[..., :-1, :].contiguous()
        shiftedIds_ofAllChoices = batch["flattened_answer_choices_ids"][
            ..., 1:
        ].contiguous()
        shiftedMasks_ofAllChoices = batch["flattened_answer_choices_mask"][
            ..., 1:
        ].contiguous()

        maxLen_ofAnswerChoices = shiftedLogits_ofAllChoices.shape[1]
        vocab_size = shiftedLogits_ofAllChoices.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x num_choices x (max_choice_len-1)]
        logProb_ofAnswerChoicesIds = -F.cross_entropy(
            shiftedLogits_ofAllChoices.view(-1, vocab_size),
            shiftedIds_ofAllChoices.view(-1),
            reduction="none",
        )

        (
            logProb_ofAnswerChoices,
            logProb_ofAnswerChoiceIds_zeroOutPadIds,
            answerChoices_len,
        ) = computeLogProb_perChoice(
            logProb_ofAnswerChoicesIds,
            shiftedMasks_ofAllChoices,
            batch["non_null_answer_choices"],
            num_ofAnswerChoices,
            maxLen_ofAnswerChoices,
            length_normalization,
        )

        _, predicted_choice = torch.max(logProb_ofAnswerChoices, dim=1)

        return (
            predicted_choice.cpu().numpy().tolist(),
            round_nestedList(logProb_ofAnswerChoices.cpu().numpy().tolist(), 5),
            round_nestedList(
                logProb_ofAnswerChoiceIds_zeroOutPadIds.cpu().numpy().tolist(), 4
            ),
            answerChoices_len.cpu().numpy().tolist(),
        )

    def generate(
        self,
        batch: Dict,
        max_generationLength: int,
        sample_tokens: bool,
        input_key: str = "",
        useFor_entropyMinimization: bool = False,
    ) -> str:
        """
        Args:
            batch:
            max_generationLength:
            sample_tokens:
            input_key: whether to use a different input besides input_ids with a different prefix
            useFor_entropyMinimization:

        Returns:
        """

        if sample_tokens:
            raise ValueError("Sampling tokens not implemented yet")

        # Use self implementation of greedy decoder to force all the ids
        # after the eos_token to also be eos tokens. This makes it easier
        # when doing generation for entropy minimization
        generation_output = greedyGeneration_decoder(
            self.transformer,
            batch[f"{input_key}input_ids"],
            batch[f"{input_key}input_mask"],
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            max_generationLength,
        )

        # generation_output = self.transformer.generate(
        #     input_ids=batch[f"{input_key}input_ids"],
        #     attention_mask=batch[f"{input_key}input_mask"],
        #     max_new_tokens=max_generationLength,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     bos_token_id=self.tokenizer.bos_token_id,
        #     do_sample=sample_tokens,
        #     return_dict_in_generate=True,
        # )

        # Remove the original input ids from the generated ids to get just the
        input_len = batch[f"{input_key}input_ids"].shape[-1]

        generated_ids = generation_output["sequences"][:, input_len:]

        generated_txt = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if useFor_entropyMinimization:
            generated_scores = torch.cat(generation_output["scores"], dim=1)
            generated_ids = generation_output["sequences"][:, 1:]
            generatedIds_mask = generated_ids != self.tokenizer.pad_token_id
            return generated_scores, generatedIds_mask
        else:
            return generation_output["sequences"].cpu().numpy().tolist(), generated_txt

    def tokenize_fn(self, batch_ofDatapoints: List[Dict], train_or_eval: str, device):
        """

        Args:
            batch_ofDatapoints:
            train_or_eval:
            device:

        Returns:

        """
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            # Construct decoder_input for training by combining input and target
            if train_or_eval == "train":
                # Assume the decoder tokenizer adds a BOS token but not EOS token, so we add the EOS token ourselves for the target
                datapoint["decoder_input"] = datapoint["input"] + datapoint["target"]

            # Gather together all the values per key
            for key, value in datapoint.items():
                if key in datapoint_batched:
                    datapoint_batched[key].append(value)
                else:
                    datapoint_batched[key] = [value]

        if train_or_eval == "train":
            # The target for training will be retrieved from decoder_input, so decoder_input has to add EOS. The target is kept to know the mask for which ids to actually compute the loss for.
            # EOS token is not added to target since the target is just used as a mask, and it has an extra BOS token from the tokenizer
            # The padding_side is right for the input and left for the target
            keys_toTokenize = [
                ("decoder_input", "decoder_input_tokenizer"),
                ("target", "target_tokenizer"),
            ]
        else:
            # For inference, the input without the target is needed
            # The padding_side is right for the input and left for the target
            keys_toTokenize = [("input", "input_tokenizer")]

            if "answer_choices" in datapoint_batched:
                # Flatten answer choices from list of list to list so that it can be tokenized
                datapoint_batched["flattened_answer_choices"] = flatten_list(
                    datapoint_batched["answer_choices"]
                )
                # The padding_side is right for the input and left for the target.
                keys_toTokenize.append(("flattened_answer_choices", "target_tokenizer"))

        # Tokenize keys which should be tokenized
        for key, tokenizer_type in keys_toTokenize:
            if tokenizer_type == "input_tokenizer":
                tokenizer = self.input_tokenizer
            elif tokenizer_type == "target_tokenizer":
                tokenizer = self.target_tokenizer
            else:
                assert tokenizer_type == "decoder_input_tokenizer"
                tokenizer = self.decoderInput_tokenizer

            tokenized_dict = tokenizer(
                datapoint_batched[key],
                return_tensors="pt",
                padding="longest",
                truncation="longest_first",
            )

            input_ids = tokenized_dict["input_ids"]
            attention_mask = tokenized_dict["attention_mask"]

            if device is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            datapoint_batched[f"{key}_ids"] = input_ids
            datapoint_batched[f"{key}_mask"] = attention_mask

        return datapoint_batched
