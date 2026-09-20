# Indic24 Qwen3-ASR pilot workflow

The raw manifests contain 25 language categories. The current audited pilot
uses 24: Hinglish is quarantined because all of its rows come from the same
manually observed misaligned source. Only English and Hindi are native
languages of the base Qwen3-ASR checkpoint; the other labels are fine-tuning
extensions.

## 1. Environment

Use a fresh environment. The repository pins Transformers 4.57.6; newer major
versions are not compatible with this checkout.

```bash
uv venv --python /home/rdp/.local/bin/python3.12 .venv-qwen3-asr
source .venv-qwen3-asr/bin/activate
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -e . datasets peft jiwer
```

The initial smoke run uses PyTorch SDPA and does not require FlashAttention.
Install FlashAttention only after the smoke path works, then pass
`--attn-implementation flash_attention_2` to training/evaluation.

Confirm that `nvidia-smi` works before model inference or training.

## 2. Prepared manifests

The generated files are under `../processed_data/qwen3_asr_indic25/`:

- `train.qwen.jsonl`: 4,309,032 rows
- `validation.qwen.jsonl`: 163,246 rows after exact transcript-overlap removal
- `validation_balanced.qwen.jsonl`: 200 rows per language, 5,000 total
- `train_smoke_balanced.qwen.jsonl`: 200 rows per language, 5,000 total
- `preparation_report.json`: reproducible counts and paths
- `train_quality_audit.json`: transcript-density audit by language and source
- `train_pilot_audited.qwen.jsonl`: 500 rows per eligible language, 12,000 total
- `validation_audited_smoke.qwen.jsonl`: 10 rows per eligible language, 240 total
- `validation_audited.qwen.jsonl`: 100 rows per eligible language, 2,400 total

Do not train directly on `train.qwen.jsonl` yet. The audit quarantines
`adjaysagar/nirantar` because its transcript-density distribution is impossible
across languages, and `adjaysagar/tts-superb-data-hindi` because manual ASR
checks found fluent audio paired with unrelated Hinglish references. The
audited subset also limits `adjaysagar/high-quality-tts` to 25% per language.

Re-run the audit and subset construction with:

```bash
python finetuning/audit_manifest_quality.py \
  ../processed_data/train.jsonl \
  --output ../processed_data/qwen3_asr_indic25/train_quality_audit.json

python finetuning/build_audited_subset.py \
  ../processed_data/qwen3_asr_indic25/train.qwen.jsonl \
  --output ../processed_data/qwen3_asr_indic25/train_pilot_audited.qwen.jsonl \
  --report ../processed_data/qwen3_asr_indic25/train_pilot_audited.report.json \
  --target-per-language 500 \
  --exclude-source adjaysagar/nirantar \
  --exclude-source adjaysagar/tts-superb-data-hindi \
  --max-synthetic-fraction 0.25
```

Regenerate them from the workspace root with:

```bash
python finetuning/prepare_multilingual_data.py \
  --workspace-root .. \
  --train-manifest ../processed_data/train.jsonl \
  --validation-manifest ../processed_data/val.jsonl \
  --output-dir ../processed_data/qwen3_asr_indic25 \
  --min-duration 0.5 \
  --max-duration 30 \
  --balanced-eval-per-language 200
```

Transcript-level filtering improves the inherited validation split but does not
prove speaker independence. A genuine-call, speaker-disjoint test set is still
required for final reporting.

## 3. Audited pilot baseline

Run automatic language identification on the 240-row gate before and after
training. Use the 2,400-row audited validation only after the gate passes.

```bash
python finetuning/evaluate_multilingual_asr.py \
  --manifest ../processed_data/qwen3_asr_indic25/validation_audited_smoke.qwen.jsonl \
  --predictions runs/audited-baseline-auto/predictions.jsonl \
  --metrics-output runs/audited-baseline-auto/metrics.json \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --language-mode auto \
  --batch-size 8 \
  --max-new-tokens 256 \
  --attn-implementation sdpa
```

## 4. RPCA channel extraction

The implementation selects a quiet 10-second region from each recording. The
paper's literal lambda, `0.5 / max(shape)`, produced an all-sparse decomposition
and silent channels on these recordings. The operational default is
`0.5 / sqrt(max(shape))`; the literal paper value can still be supplied using
`--lam` for replication.

```bash
python finetuning/telephony_augmentation.py extract \
  call_recordings/*.wav call_recordings/*.mp3 \
  --output-dir call_recordings/rpca_channels \
  --segment-seconds 10
```

Listen to and transcribe every extracted characteristic before training. Reject
any file containing intelligible residual speech. The synthetic bandpass,
8-kHz round-trip, and mu-law path remains available without RPCA files.

## 5. Audited clean LoRA pilot

The 20-step corrected smoke test passed the engineering gate but was too short
to improve recognition reliably. Run a 300-step clean pilot next. This sees
9,600 examples (80% of the 12,000-row pilot at effective batch size 32) and
keeps augmentation out of the comparison.

```bash
python finetuning/qwen3_asr_multilingual_sft.py \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --train-file ../processed_data/qwen3_asr_indic25/train_pilot_audited.qwen.jsonl \
  --eval-file ../processed_data/qwen3_asr_indic25/validation_audited_smoke.qwen.jsonl \
  --output-dir runs/indic24-lora-clean-pilot-300 \
  --batch-size 4 \
  --grad-acc 8 \
  --max-steps 300 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --language-alpha 0.5 \
  --augmentation-probability 0 \
  --log-steps 10 \
  --eval-steps 50 \
  --save-steps 50 \
  --save-total-limit 2 \
  --num-workers 2 \
  --attn-implementation sdpa
```

Evaluate `checkpoint-300` on `validation_audited_smoke.qwen.jsonl` using the
same baseline command plus `--adapter-path
runs/indic24-lora-clean-pilot-300/checkpoint-300`. Proceed to the 2,400-row
validation only if macro WER/CER and LID improve without a major English/Hindi
regression. Telephony augmentation comes after this clean-data gate.

## 6. LID-aware continuation

The 2,400-row evaluation showed strong forced-language transcription gains but
only 35% exact automatic LID, with English/Hindi auto-mode regression. Continue
from the 300-step adapter using a fresh, lower-rate schedule. Weight the three
generated metadata tokens (`language X<asr_text>`) fourfold and sample English
and Hindi twice as often as each extension language. This is a conservative
proxy for an auxiliary utterance-level LID objective while retaining the
released Qwen output format.

```bash
python finetuning/qwen3_asr_multilingual_sft.py \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --init-adapter runs/indic24-lora-clean-pilot-300/checkpoint-300 \
  --train-file ../processed_data/qwen3_asr_indic25/train_pilot_audited.qwen.jsonl \
  --eval-file ../processed_data/qwen3_asr_indic25/validation_audited_smoke.qwen.jsonl \
  --output-dir runs/indic24-lora-lid-aware-stage2-150 \
  --batch-size 4 \
  --grad-acc 8 \
  --max-steps 150 \
  --learning-rate 3e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --language-alpha 0.5 \
  --language-weights English=2,Hindi=2 \
  --language-token-weight 4 \
  --augmentation-probability 0 \
  --log-steps 10 \
  --eval-steps 50 \
  --save-steps 50 \
  --save-total-limit 3 \
  --num-workers 2 \
  --attn-implementation sdpa
```

Evaluate checkpoints 50, 100, and 150 first on the 240-row auto-language gate.
Select by macro WER/CER, exact LID, and English/Hindi retention—not eval loss
alone. Only the selected checkpoint advances to the 2,400-row evaluation.

## 7. Experiment gates

1. Baseline the unmodified model.
2. Run clean-only LoRA.
3. Run LoRA with synthetic telephony augmentation.
4. Run LoRA with reviewed RPCA plus synthetic augmentation.
5. Compare per-language macro WER/CER and language-ID accuracy on clean,
   synthetic-telephone, and genuine-call test sets.
6. Consider full fine-tuning only if LoRA saturates. Invoke it with
   `--full-finetune` and lower the default learning rate to approximately
   `2e-5`.

Forced alignment is not a 25-language evaluation path: the released aligner
supports only 11 languages.
