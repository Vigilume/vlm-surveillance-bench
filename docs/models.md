# Models

The vision-language models (VLMs) we evaluate on our dataset (45 frames + 30 clips), and how each one is run.

## Summary

| # | Model | Family | Size | Type | How we run it | Hardware | Images | Clips |
|---|---|---|---|---|---|---|---|---|
| 1 | `qwen3.5:9b` | Qwen (Alibaba) | 9B | open weights | Ollama | Kaggle, 1× T4 | ✅ | via sampled frames |
| 2 | `qwen3.5:27b` | Qwen (Alibaba) | 27B | open weights | Ollama | Kaggle, 2× T4 (split) | ✅ | via sampled frames |
| 3 | `InternVL3_5-8B` | InternVL (Shanghai AI Lab) | 8.5B | open weights | Hugging Face `transformers` | Kaggle, 2× T4 | ✅ | via sampled frames |
| 4 | `gemma4:12b` | Gemma (Google) | 12B | open weights | Ollama | Kaggle, 1× T4 | ✅ | via sampled frames |
| 5 | `llama3.2-vision:11b` | Llama (Meta) | 11B | open weights | Ollama | Kaggle, 1× T4 | ✅ | via sampled frames |
| 6 | Gemini 3.5 Flash | Gemini (Google) | – | API, free tier | Google AI Studio API | cloud | ✅ | ✅ native video |
| 7 | ChatGPT Plus (image upload) | GPT (OpenAI) | – | web app, paid subscription | manual, through the ChatGPT website (Ali) | cloud | ✅ | via uploaded frames |

**Size tiers:** medium (7–12B): #1, #3, #4, #5 · large (>24B): #2 · closed / upper bound: #6, #7.

## Running the Ollama models on Kaggle

All Ollama models in this list (#1 `qwen3.5:9b`, #2 `qwen3.5:27b`, #4 `gemma4:12b`, #5 `llama3.2-vision:11b`) use the same notebook as a template: [`Experiments/kaggle-with-ollama.ipynb`](../Experiments/kaggle-with-ollama.ipynb).

1. Open the notebook on Kaggle with **Accelerator → GPU T4 x2** and **Internet → On**.
2. Change **only the model name** in the download step:
   ```python
   MODEL = "qwen3.5:9b"   # or "qwen3.5:27b", "gemma4:12b", "llama3.2-vision:11b"
   ```
3. Run all cells. The notebook already splits the model over both GPUs (`OLLAMA_SCHED_SPREAD=1`), which `qwen3.5:27b` needs.

**Comparisons this set allows:**
- **Same family, different size:** `qwen3.5:9b` vs `qwen3.5:27b`, to see what 3× more parameters buys.
- **Five open families at a similar size (8–12B):** Qwen, InternVL, Gemma, Llama.
- **Open vs closed:** the best open model against Gemini and ChatGPT.

## Models

### 1. Qwen3.5-9B — `qwen3.5:9b`
- Natively multimodal: text, images and video in one model (no separate "VL" version in the Qwen3.5 generation).
- Ollama's default version is 4-bit quantized; the download is about 6.5 GB (≈ 5.6 GB weights + ≈ 0.9 GB vision part).
- It is a *thinking* model: call it with `think=False` for direct answers, or answers can take minutes (the notebook's `ask()` helper already does this).

### 2. Qwen3.5-27B — `qwen3.5:27b`
- Same family as #1, three times larger.
- About 17–19 GB in 4-bit, which does **not** fit one T4 (≈ 15 GB usable). The Kaggle notebook template already sets `OLLAMA_SCHED_SPREAD=1`, so Ollama splits it over both T4s (≈ 29 GB total).
- Expect slower answers than #1; measure latency separately.

### 3. InternVL3.5-8B — `OpenGVLab/InternVL3_5-8B`
- Not in Ollama's official library: run with Hugging Face `transformers` (follow the model card's loading code).
- Released in BF16. Kaggle T4s have no fast BF16, so load in **FP16** (≈ 17 GB, split over both T4s with `device_map="auto"`) or in **8-bit / 4-bit** (bitsandbytes) on one T4. Record which one was used.
- A different family from Qwen, so it gives an independent comparison at the same size.

### 4. Gemma 4 12B — `gemma4:12b`
- Google's 2026 open model family (sizes E2B, E4B, 12B, 26B, 31B), Apache 2.0.
- 4-bit fits one T4.
- **To verify:** the exact Ollama tag, and whether the 12B accepts video. Sources disagree; we send sampled frames either way.

### 5. Llama 3.2 Vision 11B — `llama3.2-vision:11b`
- Meta's vision model; **images only**. For clips we send sampled frames.
- Older (2024) than the other open models; serves as a well-known baseline.
- 4-bit fits one T4.

### 6. Gemini 3.5 Flash — API, free tier
- Closed model through Google AI Studio. The free tier has rate limits per minute and per day; read the current limits in AI Studio rather than relying on fixed numbers.
- The only model in the list with **native video input** (a whole clip, not sampled frames).
- ⚠️ **Data policy:** free-tier inputs may be used by Google to improve its products. Our samples contain faces and copyrighted footage, so send only samples we are allowed to share, or use the paid tier.

### 7. ChatGPT Plus with image upload — run manually by Ali
- Used through the **ChatGPT website**, not the API: images are uploaded and prompts typed by hand.
- Because it is manual:
  - copy the exact prompt from `prompts/` every time, and paste the answer into the results sheet unchanged;
  - write down the **model name shown in ChatGPT** and the date (the model behind ChatGPT changes over time);
  - **latency is not comparable** with the other models (human timing, network, web interface), so it is excluded from the efficiency comparison;
  - start a **new chat for every sample**, so earlier images do not influence the answer.
- ⚠️ Check ChatGPT's data settings (whether chats are used for training) before uploading our frames.

## How clips are given to the models

Ollama and most open models accept **images, not video files**. For every clip we:
1. sample **N frames** evenly from the clip (same N for every model; start with N = 8),
2. send them as multiple images with the same prompt,
3. record N in the results.

Gemini (#6) also gets the full clip as video, to compare "sampled frames" against "native video".

## Fairness rules

- **Same prompt** (stored in `prompts/`), same images, same frame sampling for every model.
- **Record the precision** (4-bit, 8-bit, FP16) and the exact model tag or version with every result.
- **Measure latency on the same GPU type** (Kaggle T4) for all open models, with a warm-up run first.
- **Save raw answers** for every sample, not only scores, so metrics can be recomputed later.

## Note on the LLM judge

The judge that grades answers must come from a **different family** than the models being tested, to avoid self-preference bias. This list already uses Qwen, InternVL, Gemma, Llama, Gemini and GPT, so the judge should be from another family, for example **Claude** (Anthropic) or **Mistral**.

## Open items

- [ ] Verify the Ollama tags `gemma4:12b` and `qwen3.5:27b` with `ollama pull` on Kaggle.
- [ ] Decide N (frames per clip) and the image resolution.
- [ ] Check Gemini free-tier limits and ChatGPT data settings before uploading samples.
