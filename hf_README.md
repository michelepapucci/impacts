---
language:
- it
license: cc-by-4.0
configs:
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
- config_name: wikipedia
  data_files:
  - split: train
    path: wikipedia/train-*.parquet
- config_name: public_administration
  data_files:
  - split: train
    path: public_administration/train-*.parquet
- config_name: all_profiling
  data_files:
  - split: train
    path: all_profiling/train-*.parquet
- config_name: wikipedia_profiling
  data_files:
  - split: train
    path: wikipedia_profiling/train-*.parquet
- config_name: public_administration_profiling
  data_files:
  - split: train
    path: public_administration_profiling/train-*.parquet
task_categories:
- text-generation
- translation
tags:
- text-simplification
- legal
- wikipedia
pretty_name: IMPaCTS
size_categories:
- 1M<n<10M
---

# IMPaCTS: Italian Multi-level Parallel Corpus for Controlled Text Simplification

IMPaCTS is a large-scale Italian parallel corpus for controlled text simplification, containing complex–simple sentence pairs automatically generated using Large Language Models. Each pair is annotated with readability scores (via Read-IT; paper [here](https://aclanthology.org/W11-2308.pdf)) and a rich set of linguistic features obtained with ProfilingUD (paper [here](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.883.pdf), web-based tool [here](http://www.italianlp.it/demo/profiling-UD/)).
The dataset is a cleaned subset of the dataset constructed for the LREC2026 paper, containing 1.066.828 pairs.


## Dataset Summary

| Split | # Sentence Pairs | # Original Sentences |
|---|---|---|
| `wikipedia` | 764061 | 106680 |
| `public_administration` | 302767 | 39820 |
| `all` (combined) | 1066828 | 146500 |

Average number of simplifications per original sentence: **9.6**

## Dataset Structure

Each row represents a (complex sentence, simplified sentence) pair.

### Key Columns

| Column | Type | Description |
|---|---|---|
| `idx` | int | Unique row identifier |
| `original_sentence_idx` | int | Unique identifier for the original sentence (multiple rows share the same original) |
| `original_text` | string | The original complex sentence (Italian) |
| `simplification` | string | The machine-generated simplified sentence |
| `domain` | string | Source domain: `wikipedia` or `public_administration` |

### Readability Scores (Read-IT)

Four scores are provided for both the original human-written texts and the automatically generated simplifications:

| Suffix | Description |
|---|---|
| `_base` | Raw textual features (avg. characters/token, avg. tokens/sentence, etc.) |
| `_lexical` | Lexical features (Type-Token Ratio, etc.) |
| `_syntax` | Syntactic features (tree depth, subordinate clause distribution, etc.) |
| `_all` | Overall readability score combining all features |

### Linguistic Features

Hundreds of additional linguistic features are provided for both sentences, with suffix `_original` (e.g., `char_per_tok_original`) or `_simplification`. These include morphological, lexical, and syntactic statistics extracted using ProfilingUD.

## Example

```python
from datasets import load_dataset

# Load the public administration domain
ds = load_dataset("mpapucci/impacts", "all")

# Or load a specific split:
# ds = load_dataset("mpapucci/impacts", split="wikipedia")
# ds = load_dataset("mpapucci/impacts", split="public_administration")

# Get all simplifications for a given original sentence
original_id = 110992
pairs = [r for r in ds if r["original_sentence_idx"] == original_id]
pairs_sorted = sorted(pairs, key=lambda x: x["simplification_all"], reverse=True)

print("Original:", pairs_sorted[0]["original_text"])
for p in pairs_sorted[1:]:
    print(f"  Readability {p['simplification_all']:.3f}:", p["simplification"])
```

## Usage

This dataset is suited for:
- Training and evaluating **text simplification** models for Italian
- **Controlled text generation** conditioned on readability scores
- Studying the effect of **linguistic features** on readability

## Citation

If you use IMPaCTS, please cite:

```bibtex
@inproceedings{papucci-etal-2026-controllable,
    title = "Controllable Sentence Simplification in {I}talian: Fine-Tuning Large Language Models on Automatically Generated Resources",
    author = "Papucci, Michele  and
      Venturi, Giulia  and
      Dell{'}Orletta, Felice",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    year = "2026",
}
```

If you use or were inspired by the dataset creation pipeline, also cite:

```bibtex
@inproceedings{papucci-etal-2025-generating,
    title = "Generating and Evaluating Multi-Level Text Simplification: A Case Study on {I}talian",
    author = "Papucci, Michele  and
      Venturi, Giulia  and
      Dell{'}Orletta, Felice",
    editor = "Bosco, Cristina  and
      Jezek, Elisabetta  and
      Polignano, Marco  and
      Sanguinetti, Manuela",
    booktitle = "Proceedings of the Eleventh Italian Conference on Computational Linguistics (CLiC-it 2025)",
    month = sep,
    year = "2025",
    address = "Cagliari, Italy",
    publisher = "CEUR Workshop Proceedings",
    url = "https://aclanthology.org/2025.clicit-1.82/",
    pages = "870--885",
    ISBN = "979-12-243-0587-3"
}
```

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).