# Extending Gender Bias Measurement in Wikipedia Corpora to South Asian Languages

This repository contains the code, data processing pipeline, and paper for extending
the gender bias measurement methodology of Chen et al. (2021), *"Gender Bias and
Under-Representation in Natural Language Processing Across Human Languages"* (AIES '21),
to South Asian languages not covered in the original 9-language study: **Bengali** and
**Hindi**.

**Bengali paper submitted to:** WiNLP 2026 (Workshop on Widening NLP), co-located with EMNLP 2026.
**Hindi analysis:** completed as a follow-up extension.

## Summary

For each language, we translate and validate the original defining set (7 word pairs)
and profession set (32 words), build a Wikipedia-based corpus, train a Word2Vec model,
and compute profession-level and corpus-level gender bias scores using the DirectBias
metric (Bolukbasi et al., 2016).

**Bengali's** overall corpus-level bias is nearly neutral under a simple average
(-0.002) but shifts to mildly female-leaning under a frequency-weighted average
(+0.027) — a pattern previously observed only in Spanish among the original nine
languages studied.

**Hindi's** corpus-level bias is also nearly neutral under a simple average (-0.002)
but, unlike Bengali, *remains* mildly male-leaning under a frequency-weighted average
(-0.012) — consistent with the majority pattern in the original nine languages.
Notably, despite having full grammatical gender (unlike Bengali), Hindi's gender
direction reliability (PCA gap: 0.072) is *weaker* than Bengali's (0.134), close to
the original paper's flagged-unreliable Chinese result (0.06). Hindi's results, which
show a striking category-level reversal compared to Bengali, should be interpreted
with appropriate caution given this reliability concern.

## Repository Structure

```
├── paper/                       # LaTeX source and compiled PDF (Bengali paper)
│   ├── acl_latex.tex
│   ├── custom.bib
│   └── final_paper.pdf
├── data_processing/
│   ├── bengali_gender_bias_pipeline.py   # Reusable, language-agnostic pipeline
│   └── hindi_gender_bias_pipeline.py     # Hindi-adapted version of the pipeline
├── results/
│   ├── bengali/
│   │   ├── bias_scores.json           # Per-profession DirectBias scores
│   │   ├── final_summary.json         # Corpus-level summary statistics
│   │   └── word_count_results.json    # Word frequency counts
│   └── hindi/
│       ├── bias_scores.json           # Per-profession DirectBias scores
│       ├── final_summary.json         # Corpus-level summary statistics
│       └── word_count_results.json    # Word frequency counts
├── HOW_TO_EXTEND.md              # General guide for extending to any new language
└── README.md
```

## Methodology Overview

1. **Word lists**: Translate Chen et al.'s (2021) defining set and profession set into
   the target language, documenting language-specific translation ambiguities
   (overlapping everyday terms, borrowed vs. native profession words, vestigial or
   systematic gendered forms, multi-word phrases).
2. **Corpus construction**: Download the target language's Wikipedia dump, extract
   clean text using WikiExtractor.
   - Bengali: 187,772 articles, 1.2GB
3. **Model training**: Train a Word2Vec model (skip-gram, vector size 300, window 5,
   min count 5) on the full corpus.
4. **Bias calculation**: Compute the gender direction via PCA on the defining pairs,
   then calculate DirectBias scores for all profession words, using frequency-weighted
   averages for words with multiple forms (borrowed/native, gendered, or synonym).

## Language-Specific Notes

**Bengali** — largely non-gendered grammar, but 5 profession words retain vestigial
masculine/feminine forms (teacher, actor, writer, nurse, student); 6 professions have
borrowed/native variants; 2 are multi-word phrases (filmmaker, comedian).

**Hindi** — broad, systematic grammatical gender; more profession words show
masculine/feminine splits than Bengali (teacher, writer, student, actor, aide, nurse —
with nurse uniquely splitting three ways: borrowed neutral + native male + native
female); 4 professions are multi-word phrases (filmmaker, comedian, adventurer,
astronaut); commonly uses the respectful "ji" honorific suffix (पिताजी, मैडम जी) across
multiple words, a pattern not seen in Bengali or the original 9 languages.

## Reproducing This Work

The pipelines in `data_processing/` (`bengali_gender_bias_pipeline.py` and
`hindi_gender_bias_pipeline.py`) are designed to run in Google Colab (for free
GPU/CPU access and easy Google Drive integration). The Bengali script is the original,
language-agnostic base pipeline; the Hindi script adapts it with Hindi's word lists,
Wikipedia dump, and Devanagari font handling. See `HOW_TO_EXTEND.md` for a full
step-by-step guide to adapting the pipeline to any new language.

**First run (from scratch):**
```python
import bengali_gender_bias_pipeline as pipeline
pipeline.run_full_pipeline()
```

**Resuming from saved results** (after the first run, results are cached to Google Drive):
```python
model, word_counts, gender_direction, results = pipeline.resume_from_saved()
```

Note: Raw Wikipedia dumps, extracted corpora, and trained Word2Vec models are not
included in this repository due to file size. Re-running `run_full_pipeline()` will
regenerate them from the public Wikipedia dumps (e.g.,
https://dumps.wikimedia.org/bnwiki/latest/, https://dumps.wikimedia.org/hiwiki/latest/).

## Requirements

```
gensim
scikit-learn
numpy
wikiextractor
```

## Key Findings

### Bengali

| Profession | Bias Score | Direction |
|---|---|---|
| governor | -0.1408 | Male |
| analyst | -0.1338 | Male |
| scientist | -0.1271 | Male |
| musician | +0.0954 | Female |
| comedian | +0.0946 | Female |
| filmmaker | +0.0913 | Female |
| actor | +0.0912 | Female |

### Hindi

| Profession | Bias Score | Direction |
|---|---|---|
| artist | -0.1513 | Male |
| comedian | -0.1268 | Male |
| musician | -0.1032 | Male |
| governor | +0.1156 | Female |
| engineer | +0.1116 | Female |
| scientist | +0.1101 | Female |

Full results for all 32 professions per language are in `results/bengali/bias_scores.json`,
`results/hindi/bias_scores.json`, and Appendix A of the paper (Bengali). Word-frequency
counts for every target word are in each language's `word_count_results.json`.

## Future Work

- Comparison against contextual embeddings (mBERT, XLM-R, BanglaBERT, BanglishBERT)
- Extension to newspaper and literary corpora, beyond Wikipedia
- Extension of translated word lists to additional South Asian languages
- Investigation into why Hindi's PCA reliability gap is lower than Bengali's despite
  having full grammatical gender

## Limitations

Corpora reflect Wikipedia's specific register and contributor demographics rather than
general language usage. Translation choices have not yet been validated by a second
native speaker for either language. Hindi's gender direction reliability (PCA gap:
0.072) is notably weaker than Bengali's (0.134); Hindi's profession-level results
should be interpreted with appropriate caution. See the Limitations section of the
paper for full Bengali-specific details.

## Acknowledgments

This work extends the methodology of Chen et al. (2021) and builds on the broader line
of research by Wali et al. (2020) on NLP toolchain underrepresentation across languages.
