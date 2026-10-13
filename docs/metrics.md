# Metrics

How we score the descriptions the VLMs (see [models.md](models.md)) write for our 45 frames and 30 clips. Every sample has **5 human reference descriptions**; a metric compares the model's description (the **candidate**) with them (the **references**) and returns a number.

| Metric | What it measures | Needs references | Needs the image | Range | Library |
|---|---|---|---|---|---|
| [BLEU-4](#1-bleu-4) | shared 1–4-word phrases (precision) | ✅ | ❌ | 0–1 | `pycocoevalcap` |
| [ROUGE-L](#2-rouge-l) | longest in-order shared word sequence | ✅ | ❌ | 0–1 | `pycocoevalcap` |
| [CIDEr](#3-cider-cider-d) | shared phrases, weighted by how informative they are | ✅ (several) | ❌ | 0–10 | `pycocoevalcap` |
| [BERTScore](#4-bertscore) | word-by-word meaning similarity | ✅ | ❌ | ≈ 0–1 (rescaled) | `bert-score` |
| [Embedding similarity](#5-embedding-similarity) | whole-sentence meaning similarity | ✅ | ❌ | ≈ 0–1 | `sentence-transformers` |
| [CLIPScore](#6-clipscore-and-refclipscore) | how well the text fits the image itself | ❌ | ✅ | 0–2.5 | `torchmetrics` |
| [RefCLIPScore](#6-clipscore-and-refclipscore) | fit to the image **and** to the references (harmonic mean of both) | ✅ | ✅ | 0–≈1.4 | `torchmetrics` (CLIP) + own code |

**Why several metrics?** Text is hard to score for two opposite reasons:
- a **correct** description can use completely different words ("two men fight" vs "a pair of guys brawl"), and
- a **wrong** description can reuse the right words ("two men fight near a red car" when there is no car: a *hallucination*).

Word-overlap metrics (BLEU, ROUGE-L, CIDEr) catch neither problem well; meaning-based metrics (BERTScore, embeddings) fix the first; CLIPScore checks the text against the picture. Together they cover each other's blind spots.

## Notation used below

| Symbol | Meaning |
|---|---|
| $c$ | the candidate (model) description |
| $r_1, \dots, r_M$ | the $M$ reference descriptions of the same sample ($M = 5$ for us) |
| $L_c,\ L_r$ | length in words of the candidate / a reference |
| $P,\ R$ | precision (share of the candidate that matches) and recall (share of the reference that is covered) |
| $F_1 = \frac{2PR}{P+R}$ | harmonic mean of $P$ and $R$; low if either is low |
| n-gram | $n$ consecutive words, e.g. "men fight" is a 2-gram |
| $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$ | cosine similarity of two vectors: 1 = same direction (same meaning), 0 = unrelated |

All word-based metrics lowercase the text and remove punctuation first (`pycocoevalcap`'s `PTBTokenizer`, which needs Java).

---

## 1. BLEU-4

**What it measures:** how many of the candidate's 1-, 2-, 3- and 4-word phrases appear in at least one reference.

$$
\text{BLEU-4} = \text{BP} \cdot \left(p_1\, p_2\, p_3\, p_4\right)^{1/4},
\qquad
\text{BP} = \begin{cases} 1 & \text{if } L_c > r \\ e^{\,1 - r/L_c} & \text{if } L_c \le r \end{cases}
$$

$$
p_n = \frac{\sum_{g \in c} \min\!\big(\text{count}_c(g),\ \max_j \text{count}_{r_j}(g)\big)}{\sum_{g \in c} \text{count}_c(g)}
$$

- $p_n$: **clipped n-gram precision**: share of the candidate's n-grams found in a reference; a phrase counts at most as often as it appears in a single reference (so repeating "men men men" does not help).
- $(p_1 p_2 p_3 p_4)^{1/4}$: **geometric mean**: if any $p_n = 0$, BLEU is 0.
- $\text{BP}$: **brevity penalty**; $r$ is the reference length closest to $L_c$. Short answers lose score (a 3-word answer against a 6-word reference keeps $e^{-1} = 37\%$).

**Why we use it:** it is reported in almost every captioning paper, so our numbers are comparable with the literature.
**Watch out:** no synonyms (a correct paraphrase can score 0), rewards copied wording even with wrong facts, often 0 for single short sentences. Report the **corpus-level** score over all samples. **Report in the appendix; do not choose models by it.**

## 2. ROUGE-L

**What it measures:** the **longest common subsequence** (LCS): the longest list of words that appears in both texts in the same order, gaps allowed.

$$
P_{\text{lcs}} = \frac{\text{LCS}(c, r)}{L_c}, \qquad
R_{\text{lcs}} = \frac{\text{LCS}(c, r)}{L_r}, \qquad
\text{ROUGE-L} = \frac{(1+\beta^2)\, P_{\text{lcs}}\, R_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}
$$

- $\beta$: recall weight; `pycocoevalcap` uses $\beta = 1.2$ (recall counts a bit more than precision). With $\beta = 1$ this is plain $F_1$.
- With several references, the best precision and the best recall over the references are used.

**Why we use it:** standard in captioning tables; a coverage-oriented view ("did the model say what the people said, in a sensible order?") that tolerates inserted words.
**Watch out:** same blindness as BLEU to synonyms and hallucinations. **Appendix.**

## 3. CIDEr (CIDEr-D)

**What it measures:** agreement in phrases with the **consensus** of all references, where rare, informative phrases ("fight", "alley") count much more than common ones ("a", "the"). Designed for image captions with many references, like our 5 per sample.

Each sentence becomes a vector of **TF-IDF** weights, one per n-gram $g$:

$$
w(g) = \text{TF}(g) \cdot \ln\!\frac{N}{\text{df}(g)}
$$

- $\text{TF}(g)$: how often $g$ occurs in the sentence.
- $N$: number of samples (images/clips) in the dataset; $\text{df}(g)$: number of samples whose references contain $g$. A phrase in every sample's references gets weight $\ln 1 = 0$.

The score for n-grams of length $n$ (CIDEr-D, the version all libraries compute):

$$
\text{CIDEr-D}_n(c) = \frac{10}{M} \sum_{j=1}^{M} e^{-\frac{(L_c - L_{r_j})^2}{2\sigma^2}} \cdot
\frac{\min\!\big(\mathbf{g}^n(c),\, \mathbf{g}^n(r_j)\big) \cdot \mathbf{g}^n(r_j)}{\lVert \mathbf{g}^n(c) \rVert\, \lVert \mathbf{g}^n(r_j) \rVert},
\qquad
\text{CIDEr-D} = \frac{1}{4} \sum_{n=1}^{4} \text{CIDEr-D}_n
$$

- $\mathbf{g}^n(\cdot)$: TF-IDF vector over n-grams of length $n$; the fraction is a cosine similarity.
- $\min(\cdot,\cdot)$, entry by entry: **clipping**, so repeating a rare word cannot inflate the score.
- $e^{-(L_c - L_r)^2 / 2\sigma^2}$ with $\sigma = 6$: **length penalty** (a 6-word length difference keeps 61 %).
- $\times 10$: scale factor, so scores run 0–10 (10 = identical to every reference). Real scores are far below 10.

**Why we use it:** the standard captioning metric, built for multiple references; ignores filler words. **Our main word-overlap metric.**
**Watch out:**
- IDF is computed from **the whole dataset**: always score all 75 samples in one call; never compare CIDEr across datasets. With only 75 samples IDF is noisy.
- Still blind to synonyms and still rewards copied wording with invented objects.
- VLM descriptions are often much longer than the references; the length penalty then lowers scores. Ask models for 1–2 sentences, or consider CIDEr-R (a variant for long descriptions, [arXiv:2109.13701](https://arxiv.org/abs/2109.13701)).

## 4. BERTScore

**What it measures:** meaning overlap word by word. Each word gets a **contextual vector** from a BERT-family model (default for English: RoBERTa-large); each candidate word is matched with its most similar reference word, and vice versa.

$$
P_{\text{BERT}} = \frac{1}{|c|}\sum_{x \in c} \max_{y \in r} \cos(x, y), \qquad
R_{\text{BERT}} = \frac{1}{|r|}\sum_{y \in r} \max_{x \in c} \cos(x, y), \qquad
F_{\text{BERT}} = \frac{2 P_{\text{BERT}} R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

- $x \in c$, $y \in r$: word vectors of candidate and reference; $|c|, |r|$: number of words.
- Raw scores crowd near the top (unrelated English text already scores ≈ 0.8), so we report the **rescaled** score $\hat F = \frac{F - b}{1 - b}$, where $b$ is the library's stored baseline for the model (score of random unrelated sentences).
- With several references, the library takes the maximum over references (separately for P, R and F).

**Why we use it:** gives correct paraphrases fair credit while keeping word-level detail. **Our main meaning-based metric.**
**Watch out:** hallucinated objects cost only a little (unrelated words still score ≈ 0.8 raw); scores depend on the chosen model; slower than overlap metrics (use the GPU).

## 5. Embedding similarity

**What it measures:** overall meaning similarity. A sentence-embedding model turns each whole sentence into one vector; we compare the vectors by cosine and keep the best-matching reference:

$$
\text{EmbSim}(c) = \max_{j = 1 \dots M} \cos\!\big(E(c),\, E(r_j)\big)
$$

- $E(\cdot)$: the embedding model (we start with `all-MiniLM-L6-v2`, 384 numbers per sentence).
- Maximum over references: the candidate only needs to agree with one person's view (our references describe different aspects of the scene).

**Why we use it:** cheap, fast, understands paraphrases; and it is the same operation our text-search system will use, so it also tells us how *findable* a description is.
**Watch out:** one vector per sentence blurs details: wrong objects, wrong numbers ("two" vs "three men") and negations barely change the score. Scores are model-specific: **fix one embedding model for the whole project.**

## 6. CLIPScore and RefCLIPScore

**What it measures:** how well the description fits **the image itself**, using CLIP, which maps images and text into one shared vector space. It is the only metric here that looks at the pixels and needs no references.

$$
\text{CLIP-S}(c, v) = w \cdot \max\!\big(\cos(\mathbf{c}, \mathbf{v}),\ 0\big), \qquad w = 2.5
$$

$$
\text{RefCLIP-S} = \frac{2 \cdot \text{CLIP-S} \cdot s_{\text{ref}}}{\text{CLIP-S} + s_{\text{ref}}}, \qquad
s_{\text{ref}} = \max\!\Big(\max_j \cos(\mathbf{c}, \mathbf{r}_j),\ 0\Big)
$$

- $\mathbf{c}, \mathbf{v}, \mathbf{r}_j$: CLIP vectors of the candidate text, the image, and reference $j$.
- $w = 2.5$ stretches the small raw cosines (good matches are often only ≈ 0.3) into a readable range; it changes no ranking.
- RefCLIP-S is the harmonic mean of the image score and the best text-reference similarity.

**Why we use it:** a reference-free sanity check, useful when references are missing or disputed.
**Watch out:** CLIP reads at most **77 tokens** (long descriptions are cut); weak at counting, spatial relations and small objects; one image at a time, so for clips we average over the sampled frames, which cannot judge actions over time. `torchmetrics` returns $100 \cdot \max(\cos, 0)$, not $2.5 \cdot \max(\cos, 0)$: convert, and say which scale is reported.

---

## Example

Three toy descriptions of our alley-fight frame (references: "two men fight near a wall", "a man grabs another man by the neck", "two people fight in an alley at night"):

| Candidate | | BLEU-4 | ROUGE-L | CIDEr-D* |
|---|---|---|---|---|
| A | "two men fight in an alley" (correct) | 0.707 | 0.696 | 2.67 |
| B | "a pair of guys brawl beside a building" (correct paraphrase) | 0.000 | 0.147 | 0.00 |
| C | "two men fight near a red car" (invents a car) | 0.615 | 0.780 | 2.43 |

\* CIDEr-D computed on a toy dataset of 3 samples. The word-overlap metrics rank the hallucination C above the correct paraphrase B, which is why we also use meaning-based metrics.

## Reporting rules

- Score **all samples in one run** with the same library versions; save **per-sample scores** (needed for confidence intervals) and the **raw model answers**.
- Fix one model each for BERTScore, embeddings and CLIP, and write the model names next to the results.
- Ask every VLM for descriptions of similar length (e.g. "1–2 sentences") so length does not decide the scores.
- Report BLEU-4 and ROUGE-L in the appendix; CIDEr, BERTScore and embedding similarity in the main table; CLIPScore as a sanity check.

## Planned `Metrics` class

These metrics will be implemented in one class (to be added), with one method per metric and one entry point, roughly:

```python
class Metrics:
    def bleu4(self, cands, refs) -> list[float]: ...
    def rouge_l(self, cands, refs) -> list[float]: ...
    def cider(self, cands, refs) -> list[float]: ...         # needs ALL samples at once (IDF)
    def bertscore(self, cands, refs) -> list[float]: ...     # rescaled F1
    def embedding_sim(self, cands, refs) -> list[float]: ...
    def clipscore(self, cands, images) -> list[float]: ...
    def refclipscore(self, cands, images, refs) -> list[float]: ...
    def evaluate(self, cands, refs, images=None) -> dict[str, list[float]]: ...  # all of the above, per sample
```

`cands`: one model description per sample; `refs`: the 5 references per sample, in the same order.

## Sources

- BLEU: Papineni et al., 2002, [ACL P02-1040](https://aclanthology.org/P02-1040/)
- ROUGE: Lin, 2004, [ACL W04-1013](https://aclanthology.org/W04-1013/)
- CIDEr: Vedantam et al., 2015, [arXiv:1411.5726](https://arxiv.org/abs/1411.5726); CIDEr-R: [arXiv:2109.13701](https://arxiv.org/abs/2109.13701)
- BERTScore: Zhang et al., 2020, [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)
- CLIPScore: Hessel et al., 2021, [arXiv:2104.08718](https://arxiv.org/abs/2104.08718)
