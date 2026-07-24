# Extending Gender Bias Measurement in Wikipedia Corpora to Bengali

This repository contains the code, data processing pipeline, and paper for extending
the gender bias measurement methodology of Chen et al. (2021), *"Gender Bias and
Under-Representation in Natural Language Processing Across Human Languages"* (AIES '21),
to Bengali — a language not covered in the original 9-language study.

**Paper submitted to:** WiNLP 2026 (Workshop on Widening NLP), co-located with EMNLP 2026.

## Summary

We translate and validate the original defining set (7 word pairs) and profession set
(32 words) into Bengali, build a Bengali Wikipedia corpus, train a Word2Vec model, and
compute profession-level and corpus-level gender bias scores using the DirectBias metric
(Bolukbasi et al., 2016). We find that Bengali's overall corpus-level bias is nearly
neutral under a simple average (-0.002) but shifts to mildly female-leaning under a
frequency-weighted average (+0.027) — a pattern previously observed only in Spanish
among the original nine languages studied.

## Repository Structure

```
├── paper/                       # LaTeX source and compiled PDF of the paper
│   ├── acl_latex.tex
│   ├── custom.bib
│   └── final_paper.pdf
├── data_processing/
│   └── bengali_gender_bias_pipeline.py   # Full reproducible pipeline
├── results/
│   ├── bias_scores.json               # Per-profession DirectBias scores
│   ├── final_summary.json             # Corpus-level summary statistics
│   └── word_count_results.json        # Word frequency counts for all target words
└── README.md
```

## Methodology Overview

1. **Word lists**: Translated Chen et al.'s (2021) defining set and profession set into
   Bengali, documenting Bengali-specific translation ambiguities (overlapping everyday
   terms, borrowed vs. native profession words, vestigial gendered forms, multi-word
   phrases).
2. **Corpus construction**: Downloaded the Bengali Wikipedia dump, extracted clean text
   using WikiExtractor, resulting in a 187,772-article, 1.2GB corpus.
3. **Model training**: Trained a Word2Vec model (skip-gram, vector size 300, window 5,
   min count 5) on the full corpus.
4. **Bias calculation**: Computed the gender direction via PCA on the 7 defining pairs,
   then calculated DirectBias scores for all 32 profession words, using frequency-weighted
   averages for words with multiple forms (borrowed/native or gendered).

## Reproducing This Work

The full pipeline is in `data_processing/bengali_gender_bias_pipeline.py`, designed to
run in Google Colab (for free GPU/CPU access and easy Google Drive integration).

**First run (from scratch):**
```python
import bengali_gender_bias_pipeline as pipeline
pipeline.run_full_pipeline()
```

**Resuming from saved results** (after the first run, results are cached to Google Drive):
```python
model, word_counts, gender_direction, results = pipeline.resume_from_saved()
```

Note: The raw Wikipedia dump, extracted corpus, and trained Word2Vec model are not
included in this repository due to file size (dump ~500MB, corpus ~1.2GB). Re-running
`run_full_pipeline()` will regenerate them from the public Bengali Wikipedia dump at
https://dumps.wikimedia.org/bnwiki/latest/.

## Requirements

```
gensim
scikit-learn
numpy
wikiextractor
```

## Key Findings

| Profession | Bias Score | Direction |
|---|---|---|
| governor | -0.1408 | Male |
| analyst | -0.1338 | Male |
| scientist | -0.1271 | Male |
| musician | +0.0954 | Female |
| comedian | +0.0946 | Female |
| filmmaker | +0.0913 | Female |
| actor | +0.0912 | Female |

Full results for all 32 professions are in `results/bias_scores.json` and Appendix A of the paper.

## Citation

If you use this work, please cite:

```bibtex
@misc{ahamed2026bengali,
  title={Extending Gender Bias Measurement in Wikipedia Corpora to Bengali},
  author={Ahamed, Ashik and Matthews, Jeanna},
  year={2026},
  note={Submitted to WiNLP 2026}
}
```

And the original methodology paper this work extends:

```bibtex
@inproceedings{chen2021gender,
  title={Gender bias and under-representation in natural language processing across human languages},
  author={Chen, Yan and Mahoney, Christopher and Grasso, Isabella and Wali, Esma and Matthews, Abigail and Middleton, Thomas and Njie, Mariama and Matthews, Jeanna},
  booktitle={Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society},
  pages={24--34},
  year={2021}
}
```

## Future Work

- Comparison against contextual embeddings (mBERT, XLM-R, BanglaBERT, BanglishBERT)
- Extension to Bengali newspaper and literary corpora, beyond Wikipedia
- Extension of translated word lists to Hindi

## Limitations

Our corpus reflects Wikipedia's specific register and contributor demographics rather
than Bengali usage broadly. Translation choices have not yet been validated by a second
native Bengali speaker. See the Limitations section of the paper for full details.

## Acknowledgments

This work extends the methodology of Chen et al. (2021) and builds on the broader line
of research by Wali et al. (2020) on NLP toolchain underrepresentation across languages.
