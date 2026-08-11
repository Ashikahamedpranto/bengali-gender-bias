# Extending Gender Bias Measurement to South Asian Languages and Text Domains

This repository contains the code, data processing pipelines, and paper for extending
the gender bias measurement methodology of Chen et al. (2021), *"Gender Bias and
Under-Representation in Natural Language Processing Across Human Languages"* (AIES '21),
to South Asian languages not covered in the original 9-language study — **Bengali** and
**Hindi** — and, for Bengali, to text domains beyond Wikipedia.

**Bengali Wikipedia paper submitted to:** WiNLP 2026 (Workshop on Widening NLP), co-located with EMNLP 2026.
**Bengali three-domain extension (Wikipedia + Literature + Newspaper):** in preparation for GeBNLP 2026 (Workshop on Gender Bias in NLP), co-located with AACL 2026.
**Hindi analysis:** completed as a follow-up extension (Wikipedia only).

## Summary

For each corpus, we translate and validate the original defining set (7 word pairs)
and profession set (32 words), build a text corpus, train a Word2Vec model, and
compute profession-level and corpus-level gender bias scores using the DirectBias
metric (Bolukbasi et al., 2016).

**Bengali Wikipedia's** overall corpus-level bias is nearly neutral under a simple
average (-0.0022) but shifts to mildly female-leaning under a frequency-weighted
average (+0.0268) — a pattern previously observed only in Spanish among the original
nine languages studied.

**Bengali Newspaper** (949,036 articles from Prothom Alo, 2009–2021) does *not*
replicate this flip: both the simple (-0.0139) and weighted (-0.0427) averages remain
male-leaning, and the corpus shows a balanced 16 female-leaning / 16 male-leaning
profession split. Several individual professions, including teacher, nurse, governor,
and scientist, reverse direction entirely between Wikipedia and newspaper text.

**Bengali Literature** (built from Wikisource, 96,365 proofread pages) produces an
unreliable gender direction (PCA gap: 0.049, below the original paper's own
Chinese-flagged-unreliable benchmark of 0.06), with 21 of 22 measurable professions
scoring male-leaning regardless of real-world associations — interpreted as a
breakdown of the DirectBias methodology on archaic, poetic Bengali register rather
than a genuine finding about literary bias. See `paper/` for the full discussion.

**Hindi's** corpus-level bias (Wikipedia only) is also nearly neutral under a simple
average (-0.0018) but, unlike Bengali, *remains* mildly male-leaning under a
frequency-weighted average (-0.0117) — consistent with the majority pattern in the
original nine languages. Hindi's gender direction reliability (PCA gap: 0.072) is
weaker than Bengali Wikipedia's (0.134), close to the original paper's flagged-
unreliable Chinese result (0.06).

## Repository Structure

```
├── paper/                       # LaTeX source and compiled PDFs
│   ├── acl_latex.tex
│   ├── custom.bib
│   └── final_paper.pdf
├── data_processing/
│   ├── bengali_wikipedia_pipeline.py     # Wikipedia extraction + training
│   ├── bengali_literature_pipeline.py    # Custom Wikisource Page-namespace extractor
│   ├── bengali_newspaper_pipeline.py     # Prothom Alo processing + chunked training
│   └── hindi_gender_bias_pipeline.py     # Hindi-adapted pipeline
├── results/
│   ├── bengali/
│   │   ├── wikipedia/
│   │   │   ├── bias_scores.json
│   │   │   ├── final_summary.json
│   │   │   └── word_count_results.json
│   │   ├── literature/
│   │   │   ├── bias_scores_literature.json
│   │   │   ├── final_summary_literature.json
│   │   │   └── word_counts_literature.pkl
│   │   └── newspaper/
│   │       ├── bias_scores_newspaper.json
│   │       ├── final_summary_newspaper.json
│   │       └── word_counts_newspaper.pkl
│   └── hindi/
│       ├── bias_scores.json
│       ├── final_summary.json
│       └── word_count_results.json
├── HOW_TO_EXTEND.md              # General guide for extending to any new language/domain
└── README.md
```

## Methodology Overview

1. **Word lists**: Translate Chen et al.'s (2021) defining set and profession set into
   the target language, documenting language-specific translation ambiguities.
2. **Corpus construction**: Build the target corpus from its source (Wikipedia dump,
   Wikisource dump, or newspaper archive) and extract clean text.
   - Bengali Wikipedia: 187,772 articles, 1.2GB
   - Bengali Literature: 96,365 proofread Wikisource pages, 267MB
   - Bengali Newspaper: 949,036 Prothom Alo articles (2009–2021), 4.6GB
3. **Model training**: Train a Word2Vec model (skip-gram, vector size 300, window 5,
   min count 5) on the full corpus.
4. **Bias calculation**: Compute the gender direction via PCA on the defining pairs,
   then calculate DirectBias scores for all profession words, using frequency-weighted
   averages for words with multiple forms.

## Notable Engineering Challenges Solved

- **Wikisource extraction**: standard tools (WikiExtractor) only resolve Wikipedia's
  structure; Wikisource stores actual text in a separate "Page:" namespace referenced
  by transclusion pointers. A custom parser (in `bengali_literature_pipeline.py`)
  reads the full-history dump directly and retains only proofread/validated pages.
- **Large-corpus training resilience**: the newspaper corpus's size made single-pass
  training unreliable under unstable network conditions. Training was split into
  sequential chunks with checkpointing to Google Drive after each chunk, allowing
  automatic resume from the last completed chunk after any interruption.

## Language- and Domain-Specific Notes

**Bengali** — largely non-gendered grammar, but 5 profession words retain vestigial
masculine/feminine forms; 6 professions have borrowed/native variants; 2 are
multi-word phrases. The literary corpus skews toward a narrow set of prominent,
largely pre-1960s authors (Tagore, Bankim Chandra Chattopadhyay, and others),
predominantly written in সাধু ভাষা (archaic literary register).

**Hindi** — broad, systematic grammatical gender; more profession words show
masculine/feminine splits than Bengali; commonly uses the "ji" honorific suffix,
a pattern not seen in Bengali or the original 9 languages.

## Reproducing This Work

Pipelines in `data_processing/` are designed to run in Google Colab. See
`HOW_TO_EXTEND.md` for a full step-by-step guide to adapting the pipeline to any new
language or text domain.

Note: Raw corpora and trained Word2Vec models are not included in this repository due
to file size; only final results (bias scores, summaries, and word counts) are
included directly. Large intermediate files are available on request.

## Requirements

```
gensim
scikit-learn
numpy
wikiextractor
pandas
```

## Future Work

- Comparison against contextual embeddings (mBERT, XLM-R, BanglaBERT, BanglishBERT)
- Extension of the Wikipedia/Literature/Newspaper three-domain comparison to Hindi
- Extension of translated word lists to additional South Asian languages
- Investigation into why Hindi's PCA reliability gap is lower than Bengali
  Wikipedia's despite having full grammatical gender

## Limitations

Corpora reflect each source's specific register and contributor demographics rather
than general language usage. Translation choices have not yet been validated by a
second native speaker. The literary corpus's gender direction is flagged as
unreliable; its profession-level results should be treated as illustrative of a
methodological limitation rather than a confirmed bias finding. See the paper's
Limitations section for full details.

## Acknowledgments

This work extends the methodology of Chen et al. (2021) and builds on the broader
line of research by Wali et al. (2020) on NLP toolchain underrepresentation, and
Chaloner & Maldonado (2019) on cross-domain gender bias measurement in English.
