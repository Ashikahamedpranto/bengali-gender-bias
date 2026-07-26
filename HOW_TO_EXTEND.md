# How to Extend This Gender Bias Measurement Methodology to Any Language

This guide walks through every step needed to replicate this study for a **new
language** — Hindi, Polish, Swahili, or any other language with a Wikipedia edition.
It is based on the methodology of Chen et al. (2021) and this repository's Bengali
extension, generalized so anyone can follow it from scratch.

No prior experience with NLP research is assumed, but basic Python and command-line
familiarity will help.

---

## Overview: What You're Building

By the end, you will have:
1. A translated **defining set** (7 gendered word pairs) and **profession set** (32 words) in your target language
2. A cleaned Wikipedia text corpus in that language
3. A trained Word2Vec model
4. A **gender direction** vector, computed via PCA
5. **DirectBias scores** for all 32 professions, showing whether each leans male or female
6. A corpus-level bias summary (simple average and frequency-weighted average)

---

## Step 1: Translate the Word Lists

### 1.1 The Defining Set (7 pairs)

Translate these into your target language. These define what "male" and "female"
look like mathematically:

| English |
|---|
| woman - man |
| daughter - son |
| mother - father |
| girl - boy |
| queen - king |
| wife - husband |
| madam - sir |

**Watch for these common translation problems** (all found during the Bengali
extension, and documented as known issues in the original 9-language study):

- **Overlapping words**: Some languages use the same word for "girl" and "daughter,"
  or "boy" and "son." If this happens in your language, look for a more formal or
  distinct alternative term to preserve the intended contrast — but flag this choice
  clearly, since it affects your defining set's reliability.
- **Ambiguous possessives**: In some languages (e.g., French, Spanish), possessive
  words like "her/his" agree with the grammatical gender of the *object*, not the
  *owner* — this can make certain pairs unusable. The original study removed
  "her-his" for this reason.
- **Untranslatable slang**: Casual English pairs like "gal-guy" often have no clean
  equivalent. Consider substituting more universally translatable formal pairs
  (the original study added queen-king, wife-husband, madam-sir for this reason).

### 1.2 The Profession Set (32 words)

Translate all 32 words:

```
nurse, teacher, writer, engineer, scientist, manager, driver, banker,
musician, artist, chef, filmmaker, judge, comedian, inventor, worker,
soldier, journalist, student, athlete, actor, governor, farmer, person,
lawyer, adventurer, aide, ambassador, analyst, astronaut, astronomer,
biologist
```

**For each word, check and document:**

**(a) Does your language have a borrowed (usually English-derived) form AND a
native form?**
Many languages use English loanwords for modern professions (engineer, manager,
driver, etc.) alongside older native words. If both exist and are commonly used,
keep both — you'll combine them later using a frequency-weighted average.

**(b) Does the word have separate masculine/feminine forms?**
Grammatically gendered languages (Spanish, German, Arabic, French, Urdu, and
apparently some "non-gendered" languages too — Bengali surprised us here) may have
distinct male/female forms for some professions, even if the language as a whole
doesn't mark grammatical gender broadly. Document every instance you find.

**(c) Is the word a multi-word phrase in your language?**
"Filmmaker" and "comedian," for example, are two-word phrases in several languages.
This matters for the tokenization step later (Step 4).

**Get a second native speaker to review your full translated list before proceeding**
— translation ambiguities are easy to miss on your own.

---

## Step 2: Build Your Corpus

### 2.1 Download the Wikipedia dump

Find your language's Wikipedia code (e.g., `hi` for Hindi, `pl` for Polish, `bn`
for Bengali) and download the corresponding dump:

```bash
wget https://dumps.wikimedia.org/{LANGCODE}wiki/latest/{LANGCODE}wiki-latest-pages-articles.xml.bz2
```

Replace `{LANGCODE}` with your language's code — e.g., for Hindi:
```bash
wget https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2
```

### 2.2 Extract clean text

Install and run WikiExtractor:

```bash
pip install wikiextractor
python -m wikiextractor.WikiExtractor {LANGCODE}wiki-latest-pages-articles.xml.bz2 \
    --output extracted --json --processes 4
```

**Known issue**: WikiExtractor 3.0.6 has two Python 3.12 regex compatibility bugs.
If you hit `re.error: global flags not at the start of the expression`, see the
patch described in `data_processing/bengali_gender_bias_pipeline.py` (function
`patch_wikiextractor()`) — the fix applies to any language, since it's a bug in
WikiExtractor's own code, not language-specific.

### 2.3 Combine into one clean corpus file

Read each extracted file, pull out the `"text"` field from each JSON line, and
write to a single plain-text file (one article per line, or split into
sentences). See `build_clean_corpus()` in the pipeline script for a working
example.

---

## Step 3: Train Word2Vec

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,          # your tokenized corpus
    vector_size=300,
    window=5,
    min_count=5,
    workers=4,
    sg=1                 # skip-gram
)
```

**Tokenization note**: A simple whitespace/punctuation-based tokenizer works for
most languages, but check whether your language's script needs special handling.
Chinese, for example, doesn't use spaces between words and needs a dictionary-based
tokenizer. Verify this before training by testing your tokenizer on a few sample
sentences.

---

## Step 4: Count Word Frequencies

Count how many times every word in your defining set and profession set (including
every borrowed/native and gendered variant) appears in the raw corpus text. This
is a simple frequency count — separate from the trained model — and is needed for
the weighted-average calculations in Step 6.

**For multi-word phrases**: your tokenizer likely won't capture these as a single
unit. Use a direct substring search on the raw text file instead, e.g.:
```python
full_text.count("your multi-word phrase")
```

---

## Step 5: Compute the Gender Direction

1. For each of your 7 defining pairs, calculate the vector difference:
   `diff = vector(female_word) - vector(male_word)`
2. Collect all 7 (or however many survived translation) differences into a matrix
3. Run PCA on this matrix; take the first principal component as your **gender
   direction**
4. **Check the reliability gap**: compare the explained variance of the 1st vs.
   2nd principal component. A large gap (e.g., >0.3) means your gender direction is
   trustworthy. A small gap (e.g., <0.1, as seen with Chinese in the original study)
   means the direction may be capturing something other than pure gender — report
   this honestly as a limitation.

```python
from sklearn.decomposition import PCA
import numpy as np

diffs = np.array([...])  # your collected difference vectors
pca = PCA(n_components=min(len(diffs), 10))
pca.fit(diffs)
gender_direction = pca.components_[0]
gap = pca.explained_variance_ratio_[0] - pca.explained_variance_ratio_[1]
```

---

## Step 6: Calculate DirectBias Scores

For each profession word, calculate cosine similarity between its vector and the
gender direction:

```python
def direct_bias(word_vector, gender_direction, c=1):
    cos_sim = np.dot(word_vector, gender_direction) / (
        np.linalg.norm(word_vector) * np.linalg.norm(gender_direction)
    )
    return abs(cos_sim) ** c * np.sign(cos_sim)
```

**For words with multiple forms** (borrowed/native, or gendered), calculate each
form's bias score individually, then combine using a frequency-weighted average
based on the word counts from Step 4:

```python
weighted_bias = sum(bias_i * count_i for i in forms) / sum(count_i for i in forms)
```

This matters — a simple, unweighted average can badly misrepresent the word's real
usage if one form is far more common than another (see the German "scientist"
example in Chen et al., 2021, or the "actor" example in the Bengali extension).

**For multi-word phrases**: since no single trained vector exists for the whole
phrase, approximate it by averaging the vectors of its component words before
calculating DirectBias.

---

## Step 7: Compute Corpus-Level Bias

Calculate two summary numbers across all 32 professions:

**Simple average:**
```python
simple_avg = sum(all_bias_scores) / len(all_bias_scores)
```

**Frequency-weighted average** (weight each profession by its total occurrence
count in the corpus):
```python
weighted_avg = sum(bias * count for bias, count in zip(scores, counts)) / sum(counts)
```

**Compare the two.** If they diverge in direction (one positive, one negative), this
is a genuinely interesting finding worth highlighting — it has so far only been
observed in Spanish and Bengali.

---

## Common Pitfalls Across Languages

| Issue | Where it showed up before | What to do |
|---|---|---|
| Regex bugs in WikiExtractor on Python 3.12 | Bengali | Patch as described in Step 2.2 |
| Font rendering in LaTeX for non-Latin scripts | Bengali | Use LuaLaTeX + `fontspec` + a Unicode font (e.g., Noto Sans) covering your script |
| Overleaf free-tier compile timeout with heavy font-shaping scripts | Bengali | Combine adjacent same-language text into single font-switch commands; consider a paid tier if still too slow |
| Tokenizer doesn't segment words correctly | Chinese (space-less script) | Use a language-specific tokenizer/segmenter |
| PCA doesn't cleanly isolate a gender direction | Chinese | Report the reliability gap honestly; consider adding more defining pairs |
| Wikipedia corpus too small for reliable statistics | Wolof | Report corpus size transparently; treat results as preliminary |

---

## Reporting Your Results

Whatever language you choose, aim to report at minimum:
1. Your final defining set and profession set (with translation notes)
2. Corpus size (articles, file size, sentence count)
3. The PCA reliability gap for your gender direction
4. Full per-profession DirectBias scores (simple + weighted)
5. Corpus-level bias (simple average vs. weighted average)
6. Any translation ambiguities specific to your language, framed as a contribution
   to the growing cross-linguistic taxonomy of these issues (see Section 3.1 of the
   Bengali paper in this repo for an example)

---

## Questions or Issues

If you extend this to a new language, consider opening an issue or pull request in
this repository — cross-linguistic comparisons become more valuable as more
languages are added.
