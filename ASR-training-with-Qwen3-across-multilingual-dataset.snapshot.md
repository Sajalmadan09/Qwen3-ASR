# Conversation handoff from Claude (claude.ai)

**Title:** ASR training with Qwen3 across multilingual dataset  ·  **Started:** 13 August 2026  ·  **Turns:** 5

## Briefing

You are continuing a working session that began in Claude. Read the briefing, the artefacts, and the recent turns below. Pick up where the previous assistant left off — don't re-introduce yourself, match the user's working style.

**Original ask:**
> This context is about tts training for some 5 languages but i have data of 22 languages as you can read the whole context and evrything now i have a new task to train and asr using the same dataset but the catch is in asr there a technique/reaserach being done which i am sharing reaserach paper about also the model i want to train is qwen-asr 3 0.6b repo : https://github.com/QwenLM/Qwen3-ASR read everything throughly then lets discuss. thats is codex context for dataset that we will be using

**Where the previous assistant left off:**
> ``` This block is not supported on your current device yet. ``` ``` This block is not supported on your current device yet. ``` ``` This block is not supported on your current device yet. ``` ``` This block is not supported on your current device yet. ``` ``` This block is not supported on your cur…

---

## Conversation (5 turns)

**User:**
This context is about tts training for some 5 languages but i have data of 22 languages as you can read the whole context and evrything now i have a new task to train and asr using the same dataset but the catch is in asr there a technique/reaserach being done which i am sharing reaserach paper about also the model i want to train is qwen-asr 3 0.6b repo :
https://github.com/QwenLM/Qwen3-ASR

read everything throughly then lets discuss. thats is codex context for dataset that we will be using

**Assistant:**
```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```

I've now read both documents closely, plus checked current details on Qwen3‑ASR since it launched after my training cutoff. Here's my synthesis and how I think the two pieces fit together, before we lock in a plan.

**The RPCA paper, in short.** Mo & Lam use Robust PCA (Candès' Principal Component Pursuit: minimize nuclear norm of a low‑rank part `L` plus λ times the ℓ1 norm of a sparse part `S`, subject to `L+S=M`) on spectrograms of *genuine* telephone recordings. Speech is the sparse, high‑variation component; the telephone channel (line noise, compression artifacts, background hum/music) is the low‑rank component. They build a binary time‑frequency mask from `L` vs `S`, isolate the channel-only component via inverse STFT, then mix that extracted "characteristic" onto clean, cleanly‑annotated speech at a chosen SNR to synthesize artificial telephony data — without ever needing new transcriptions, since the clean speech's existing labels stay valid. Fine-tuning/training an ASR (bLSTM‑CTC) on King ASR + this augmented clean data beat the baseline by 5.4% relative WER using just one extracted channel, and combining the augmented set + clean set + real King ASR data got a 7.8% relative WER reduction — beating even augmentation with real street noise. Interestingly, the channel characteristic that helped most (their "Characteristic 3") was the *quietest, steadiest, narrowest-bandwidth* one (lowest spectral centroid/spread/flux), not the noisiest — so it's channel coloration/compression signature that matters, not "more noise."

**Qwen3‑ASR, in short.** It's Alibaba's open-sourced (Jan 2026) all-in-one multilingual ASR family built on the Qwen3‑Omni audio encoder + LLM decoder — 0.6B and 1.7B variants, unified streaming/offline, native `transformers` support added June 2026. Officially it covers 30 languages + 22 Chinese dialects. Hindi is explicitly listed among the 30; I could **not** confirm Tamil, Telugu, or Bengali are in that official 30-language list from what's public — the model cards just say "30 languages" without printing the full table everywhere. This actually matters a lot for your plan: I found a public example (`amugoodbad229/qwen3-asr-bengali-lora`) where the base Qwen3‑ASR-1.7B **hallucinates Hindi/English characters on Bengali phonemes**, and a LoRA fine-tune on real Bengali data cut WER from a very high baseline down to ~20.7%, beating Whisper large-v3 on the same noisy multi-dialect set. That's a strong signal that for at least Bengali (and plausibly Tamil/Telugu too), Qwen3‑ASR is not a strong zero-shot performer, and fine-tuning is genuinely necessary, not just a robustness nicety.

**How the two connect for your new task, as I understand it:**
You already have a large, clean, and validated speech-text corpus (extracted from `processed_data`, with `train_filtered.jsonl` / `val_filtered.jsonl` / `dev.jsonl`, language-balanced sampling, and per-language manifests spanning up to 22 language categories, 5 of which were prioritized). Instead of using that corpus to teach VoxCPM2 to *speak* in these languages, you now want to fine-tune Qwen3‑ASR-0.6B to *transcribe* speech in these languages — and apply the RPCA telephony-augmentation trick from the paper so the resulting ASR model is robust to real telephone-channel audio (compressed, band-limited, possibly noisy calls), not just clean studio speech.

Before I sketch an implementation plan, a few things I think we need to pin down together, since each choice materially changes the pipeline:

1. **Language scope**: same 5-language subset (en/hi/ta/te/bn) we already have filtered manifests for, or do you want to reconstitute more of the original 22-language `processed_data` for this ASR run? ASR fine-tuning is much cheaper than TTS fine-tuning per sample, so going wider is more feasible here.
2. **Telephony reference recordings**: the paper needed only *five* genuine telephone recordings (any language, 3–13s each) — RPCA extracts the *channel*, not language content, so a handful of real Indian telephone/call-center recordings (landline, mobile, VoIP) would work across all languages. Do you have or can we source a few real telephony clips, or do we need to simulate channel characteristics synthetically (codec + bandpass + compression artifacts) as a fallback if no real recordings are available?
3. **Sample rate**: our corpus audio was decoded/resampled to 16kHz for VoxCPM2's encoder; Qwen3‑ASR's audio front-end likely also expects 16kHz mono — I'll confirm exact preprocessing requirements from its `qwen-asr`/`transformers` code before we build the data loader.
4. **Fine-tuning method**: given Qwen3‑ASR-0.6B is tiny compared to VoxCPM2's 2B, full fine-tuning is very plausible on your GPU, though LoRA (as in the Bengali example) is proven to work well and is faster/safer against catastrophic forgetting of the already-strong languages (English, Hindi).
5. **Augmentation recipe to replicate**: the paper's best setup wasn't "augmented only" — it was clean + King‑ASR‑real + augmented‑clean combined. We'd want to mirror that: keep clean copies of our manifests, add RPCA‑channel‑augmented copies at 0dB SNR (their best value), and train on the union — rather than fully replacing clean data.
6. **WER evaluation**: Indic-script WER needs care (word segmentation/normalization conventions differ by script, especially for Bengali/Tamil/Telugu) — do you want CER as well, given agglutinative scripts like Tamil/Telugu are often reported with CER too?

Once you tell me the scope on those, I can lay out the concrete pipeline (RPCA extraction script, augmented-manifest generator, Qwen3‑ASR data format + LoRA/full-FT training script, eval harness) the same way we did for the VoxCPM2 run.

**User:**
Answers :

1. all 22 lanaguages
2. I have 5 call recordings no worries yeah also keep fallback as well.
3. ok please confirm 
4. Sure lets go with lora then full finetune if required
5. Sure that works
6. Sure

**Assistant:**
```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```

Good — all six answered. Here's what I confirmed and what it means for the plan, then a proposed set of next steps.

**On #3 (sample rate) — confirmed.** Qwen3‑ASR's official streaming demo explicitly resamples microphone input to **16,000 Hz mono** before feeding the model, and that's consistent across the transformers/vLLM backends. That's the same target rate our VoxCPM2 pipeline already used (`Audio(sampling_rate=16000)` casting), so the underlying audio files in `processed_data/audio` don't need any new resampling step for ASR — we can point straight at them.

**On #1 (all 22 languages) — this changes the framing a bit, and it's important.** I pulled the exact official list of the 30 languages Qwen3‑ASR-0.6B natively supports:

> zh, en, yue (Cantonese), ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, **hi**, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro

Cross-referencing against your 22 target categories (English, Hindi, Hinglish, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Manipuri, Rajasthani, Kashmiri, Konkani, Nepali, Sanskrit, Santali, Sindhi, Urdu): **only English and Hindi are officially supported.** The other 20 are outside the base model's known languages entirely.

That means this isn't a uniform "fine-tune for robustness" job across all 22 — it splits into two different difficulty tiers:
- **English + Hindi**: true domain adaptation. The model already reads these; fine-tuning here is mainly about your specific data domain (Indic accents, telephony robustness via RPCA) and should converge fast, LoRA-only, likely low rank.
- **The other ~20 languages**: closer to teaching genuinely new languages to the decoder, plus new joint language-ID categories it's never seen. This is a materially bigger lift. Good news: I found a public precedent — someone LoRA-fine-tuned Qwen3‑ASR-1.7B on real Bengali (also outside the 30) and took it from a badly hallucinating base model (frequently outputting Hindi/English characters for Bengali phonemes) down to ~20.7% WER, beating Whisper-large-v3 on the same noisy set. So it's achievable via LoRA, but expect it to need more steps/higher rank than the Hindi/English tier, and per-language results will scale with how much data each language actually has in the corpus — several of the 22 (Manipuri, Kashmiri, Konkani, Sanskrit, Santali, Sindhi) were quite small in the original per-language breakdown, so I'd set realistic expectations there rather than assume uniform WER across all 22.

One open technical question I haven't nailed down yet: how the fine-tuning JSONL schema tags language, and whether forcing/teaching a brand-new language label (e.g. `ta`, `te`, `mni`, `sat`) is supported out of the box or needs a workaround. The repo does have an official `finetuning/` guide (linked from the README's Fine Tuning section), but it 502'd when I tried to open it just now — I'll fetch it properly when we get to building the training script rather than guessing at hyperparameters or the exact data format.

**On #4 (LoRA → full fine-tune) — agreed, and worth noting:** at 0.6B, this model is small enough that even full fine-tuning is very cheap on your GPU (VoxCPM2 was 2B and full-FT-adjacent LoRA already fit in ~21 GB peak). So "LoRA first" here is more about protecting the already-good English/Hindi/general-audio behavior from catastrophic forgetting than about a memory constraint — full FT stays a realistic fallback specifically for the low-resource new-language tier if LoRA rank turns out insufficient.

**On #2 and #5 — good, and they compose well together.** Since RPCA only needs the channel, not language content, your 5 real telephone recordings can be used to extract 5 (or more, via segment-splitting) channel characteristics once, then mixed onto clean utterances across all 22 languages at 0dB SNR — exactly like the paper mixed one Cantonese channel across their whole clean corpus. For the fallback, I'd build a simple synthetic-channel generator (bandpass ~300–3400Hz telephony band, μ-law/codec-style quantization, mild compression/clipping) so the pipeline doesn't hard-depend on having enough real recordings for variety.

**Given all this, here's what I'd do next, in order:**
1. Reconstruct the full 22-language manifest from `processed_data/train.jsonl` / `val.jsonl` (the deduplicated ones — ~4.3M/226K rows already clean across all 22 categories) and apply the same kind of length/quality filtering we did for the 5-language subset, sized for ASR sequence limits rather than TTS ones.
2. Fetch the actual `finetuning/` guide and confirm the JSONL schema, language-tagging mechanism, and default LoRA hyperparameters before writing any training code.
3. Build the RPCA extraction script against your 5 real recordings, plus the synthetic fallback.
4. Decide per-language data budgets/sampling temperature (mirroring the balanced sampler we used for VoxCPM2) given the 22 languages won't have anywhere near equal volume.

Want me to start with step 1 (manifest reconstruction) and step 2 (pinning down the real fine-tuning recipe) now?

**Assistant:**
```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```


```
This block is not supported on your current device yet.
```

Read `finetuning/README.md` and `finetuning/qwen3_asr_sft.py` (plus the processor code) straight from your uploaded repo. This settles the open questions and changes one part of the plan.

**Data format — confirmed and simpler than expected.** Each JSONL line is just:
```jsonl
{"audio":"/path/utt.wav","text":"language English<asr_text>This is a test sentence."}
```
The `language <Name><asr_text>` prefix is **plain text that becomes part of the model's target output**, not a special architecture token or a separate `language:` field. Looking at the collator, the `text` field gets appended after the chat-template prompt and is what gets trained as `labels` (only the prompt portion is masked to `-100`). That's genuinely good news for the 20 languages outside the official 30: teaching the model to emit `language Tamil<asr_text>...` on Tamil audio is just supervised sequence generation on new strings — no vocabulary surgery, no new embedding rows, no config changes. We just need to pick and stick with the exact display names for each language for our new categories.

**Sample rate — confirmed at the code level, not just docs.** The collator hardcodes `librosa.load(path, sr=16000, mono=True)`, and the processor's feature extractor (`WhisperFeatureExtractor`-based) also defaults to 16000 Hz. So our existing 16kHz-decoded corpus works as-is; the RPCA-augmented copies just need to be written out at 16kHz too.

**LoRA — this is the one place I need to correct myself.** The official script has **no LoRA path at all** — no `peft` import, no adapter config, nothing. `qwen3_asr_sft.py` does straight full-parameter fine-tuning: it loads `asr_wrapper.model` directly, patches its forward, and hands it to a plain HF `Trainer`. The community Bengali LoRA example I found earlier was a third-party wrapper, not this repo. So "go LoRA" here means we write the `peft.LoraConfig` wrapping ourselves around this exact pipeline (reusing the dataset loading, collator, checkpoint-copy callback unchanged) — it's a small addition, not a flag flip.

Given that, worth weighing honestly: at 0.6B, full fine-tuning is cheap (weights + grads + AdamW state in bf16/fp32 is roughly ~7 GB, trivial on our GPU) and is the officially tested path. The real argument *for* LoRA here isn't memory, it's protecting the two languages the model is already decent at (English, Hindi) from being overwritten while we aggressively teach it 20 new ones at once — a full-FT run tuned for fast new-language acquisition is a classic recipe for forgetting what already worked. So I'd still do LoRA first as you said, specifically for that reason, with full FT as the documented, zero-risk fallback using the script exactly as shipped.

**Also worth flagging:** the script's `--eval_file` only computes token-level cross-entropy loss during eval, not WER/CER. We'll need our own eval harness — load `dev.jsonl`-style held-out audio per language, call `model.transcribe()`, and score WER/CER against ground truth, the same way the RPCA paper's Table II worked.

Default hyperparams in the script (for reference, if useful as a starting point): `batch_size=32, grad_acc=4` (effective 128), `lr=2e-5`, linear schedule, `warmup_ratio=0.02`, bf16 auto-enabled on compute capability ≥8 (our GPU qualifies).

Next concrete steps I'd take, in order — want me to proceed?
1. Convert our deduplicated `processed_data/train.jsonl`/`val.jsonl` (all 22 languages) into this exact `{"audio","text"}` schema, applying length/quality filtering sized for ASR rather than TTS, and pick the language display names to use in the `language <Name><asr_text>` prefix.
2. Write the RPCA channel-extraction script against your 5 real call recordings, plus the synthetic fallback, and generate 0dB-SNR augmented copies to mix in alongside clean data.
3. Write the PEFT/LoRA wrapper around `qwen3_asr_sft.py` (target modules, rank/alpha) and a real WER/CER eval script.

---

_Continue from here. The user's next message will follow._