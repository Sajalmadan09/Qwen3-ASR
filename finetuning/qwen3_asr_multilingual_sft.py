#!/usr/bin/env python3
"""LoRA/full multilingual Qwen3-ASR trainer with telephony augmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig, TrainerCallback, TrainingArguments

try:
    from .multilingual_sampling import LanguageDurationSampler
    from .qwen3_asr_sft import (
        CastFloatInputsTrainer,
        find_latest_checkpoint,
        mask_prefix_labels,
        patch_outer_forward,
    )
    from .telephony_augmentation import resample_waveform, telephony_augment
except ImportError:  # Direct execution: python finetuning/qwen3_asr_multilingual_sft.py
    from multilingual_sampling import LanguageDurationSampler
    from qwen3_asr_sft import (
        CastFloatInputsTrainer,
        find_latest_checkpoint,
        mask_prefix_labels,
        patch_outer_forward,
    )
    from telephony_augmentation import resample_waveform, telephony_augment


def build_prefix_messages(prompt: str, audio_array: Any = None) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    return resample_waveform(waveform, int(source_rate), sample_rate)


@dataclass
class MultilingualDataCollator:
    processor: Any
    sampling_rate: int = 16000
    augmentation_probability: float = 0.0
    augmentation_mode: str = "mixed"
    snr_choices: tuple[float, ...] = (-5.0, 0.0, 5.0)
    rpca_channels: list[tuple[np.ndarray, int]] = field(default_factory=list)
    language_token_weight: float = 1.0
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            seed = worker.seed if worker is not None else torch.initial_seed()
            self._rng = np.random.default_rng(seed % (2**32))
        return self._rng

    def augment(self, waveform: np.ndarray) -> np.ndarray:
        rng = self.rng()
        if self.augmentation_probability <= 0 or rng.random() >= self.augmentation_probability:
            return waveform

        use_rpca = bool(self.rpca_channels) and self.augmentation_mode == "rpca"
        if self.augmentation_mode == "mixed" and self.rpca_channels:
            use_rpca = bool(rng.integers(0, 2))
        channel = None
        channel_rate = 8000
        if use_rpca:
            channel, channel_rate = self.rpca_channels[int(rng.integers(0, len(self.rpca_channels)))]
        return telephony_augment(
            waveform,
            sample_rate=self.sampling_rate,
            channel=channel,
            channel_sample_rate=channel_rate,
            snr_db=float(rng.choice(self.snr_choices)),
            seed=int(rng.integers(0, 2**32 - 1)),
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        targets = [str(feature["text"]) for feature in features]
        prefix_texts = []
        audios = []
        for feature in features:
            messages = build_prefix_messages(str(feature.get("prompt", "")))
            prefix = self.processor.apply_chat_template(
                [messages], add_generation_prompt=True, tokenize=False
            )[0]
            prefix_texts.append(prefix)
            audios.append(self.augment(load_audio(str(feature["audio"]), self.sampling_rate)))

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        pad_id = self.processor.tokenizer.pad_token_id
        full_inputs["labels"] = mask_prefix_labels(full_inputs, prefix_inputs, pad_id)
        if self.language_token_weight != 1.0:
            loss_weights = torch.ones_like(full_inputs["labels"], dtype=torch.float32)
            for index, target in enumerate(targets):
                marker = target.find("<asr_text>")
                if marker < 0:
                    continue
                language_target = target[: marker + len("<asr_text>")]
                language_length = len(
                    self.processor.tokenizer(language_target, add_special_tokens=False)["input_ids"]
                )
                supervised = torch.nonzero(
                    full_inputs["labels"][index].ne(-100), as_tuple=False
                ).flatten()
                loss_weights[index, supervised[:language_length]] = self.language_token_weight
            full_inputs["loss_weights"] = loss_weights
        return full_inputs


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor: Any, run_metadata: dict[str, Any]):
        self.processor = processor
        self.run_metadata = run_metadata

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(checkpoint)
        (checkpoint / "multilingual_training_config.json").write_text(
            json.dumps(self.run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return control


class MultilingualTrainer(CastFloatInputsTrainer):
    def __init__(
        self,
        *args,
        language_alpha: float,
        language_weights: dict[str, float],
        duration_boundaries: tuple[float, ...],
        **kwargs,
    ):
        self.language_alpha = language_alpha
        self.language_weights = language_weights
        self.duration_boundaries = duration_boundaries
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not hasattr(dataset, "column_names"):
            return super()._get_train_sampler(train_dataset)
        if "language" not in dataset.column_names or "duration" not in dataset.column_names:
            return super()._get_train_sampler(train_dataset)
        return LanguageDurationSampler(
            dataset["language"],
            dataset["duration"],
            batch_size=self.args.per_device_train_batch_size,
            language_alpha=self.language_alpha,
            language_weights=self.language_weights,
            duration_boundaries=self.duration_boundaries,
            seed=self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        if loss_weights is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs["labels"]
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous().to(shift_logits.device)
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        active = shift_labels.ne(-100)
        weighted = shift_weights * active
        loss = (token_losses * weighted).sum() / weighted.sum().clamp_min(1.0)
        return (loss, outputs) if return_outputs else loss


def load_rpca_channels(paths: list[Path]) -> list[tuple[np.ndarray, int]]:
    channels = []
    for path in paths:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        channels.append((waveform.mean(axis=1), int(sample_rate)))
    return channels


def comma_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def comma_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def language_weight_map(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        for item in value.split(","):
            if not item.strip():
                continue
            language, weight = item.rsplit("=", 1)
            result[language.strip().lower()] = float(weight)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected comma-separated language=weight entries") from exc
    if any(not key or weight <= 0 for key, weight in result.items()):
        raise argparse.ArgumentTypeError("language names must be nonempty and weights positive")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        type=comma_strings,
        default=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-acc", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--language-alpha", type=float, default=0.5)
    parser.add_argument("--language-weights", type=language_weight_map, default={})
    parser.add_argument("--language-token-weight", type=float, default=1.0)
    parser.add_argument(
        "--duration-boundaries",
        type=comma_floats,
        default=(2.0, 4.0, 8.0, 12.0, 20.0, 30.0),
    )
    parser.add_argument("--augmentation-probability", type=float, default=0.0)
    parser.add_argument("--augmentation-mode", choices=("synthetic", "rpca", "mixed"), default="mixed")
    parser.add_argument("--snr-choices", type=comma_floats, default=(-5.0, 0.0, 5.0))
    parser.add_argument("--rpca-channel", type=Path, action="append", default=[])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=47803)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.grad_acc <= 0:
        parser.error("batch-size and grad-acc must be positive")
    if not 0 <= args.augmentation_probability <= 1:
        parser.error("augmentation-probability must be in [0, 1]")
    if args.augmentation_mode == "rpca" and not args.rpca_channel:
        parser.error("augmentation-mode=rpca requires at least one --rpca-channel")
    if args.language_alpha < 0:
        parser.error("language-alpha must be non-negative")
    if args.language_token_weight <= 0:
        parser.error("language-token-weight must be positive")
    if args.full_finetune and args.init_adapter:
        parser.error("--init-adapter cannot be combined with --full-finetune")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-ASR fine-tuning")

    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
        attn_implementation=args.attn_implementation,
    )
    model = wrapper.model
    processor = wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if int(os.environ.get("RANK", "0")) == 0:
            model.print_trainable_parameters()
    elif not args.full_finetune:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.lora_targets),
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if int(os.environ.get("RANK", "0")) == 0:
            model.print_trainable_parameters()

    files = {"train": str(args.train_file.resolve())}
    if args.eval_file:
        files["validation"] = str(args.eval_file.resolve())
    dataset = load_dataset("json", data_files=files)

    rpca_channels = load_rpca_channels(args.rpca_channel)
    collator = MultilingualDataCollator(
        processor=processor,
        sampling_rate=args.sample_rate,
        augmentation_probability=args.augmentation_probability,
        augmentation_mode=args.augmentation_mode,
        snr_choices=args.snr_choices,
        rpca_channels=rpca_channels,
        language_token_weight=args.language_token_weight,
    )

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-5 if args.full_finetune else 1e-4
    evaluation_enabled = args.eval_file is not None
    training_args = TrainingArguments(
        output_dir=str(args.output_dir.resolve()),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_steps,
        eval_strategy="steps" if evaluation_enabled else "no",
        eval_steps=args.eval_steps if evaluation_enabled else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
    )
    run_metadata = vars(args).copy()
    run_metadata = {
        key: [str(item) for item in value] if isinstance(value, list) and value and isinstance(value[0], Path)
        else str(value) if isinstance(value, Path)
        else value
        for key, value in run_metadata.items()
    }
    trainer = MultilingualTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=collator,
        processing_class=processor,
        callbacks=[SaveProcessorCallback(processor, run_metadata)],
        language_alpha=args.language_alpha,
        language_weights=args.language_weights,
        duration_boundaries=args.duration_boundaries,
    )

    resume_from = args.resume_from.strip()
    if not resume_from and args.resume:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""
    trainer.train(resume_from_checkpoint=resume_from or None)
    trainer.save_model(training_args.output_dir)
    if trainer.args.process_index == 0:
        processor.save_pretrained(training_args.output_dir)
        (Path(training_args.output_dir) / "multilingual_training_config.json").write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
