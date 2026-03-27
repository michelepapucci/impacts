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
task_ids:
- text-simplification
tags:
- text-simplification
- legal
- wikipedia
- italian
- readability
- controllable-generation
- linguistics
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

## Available Configs

IMPaCTS comes in two variants, each available for three domain splits:

| Config | Columns |
|---|---|
| `all` | Core columns only (12) | 
| `wikipedia` | Core columns only (12) | 
| `public_administration` | Core columns only (12) | 
| `all_profiling` | Core + ~300 linguistic features | 
| `wikipedia_profiling` | Core + ~300 linguistic features | 
| `public_administration_profiling` | Core + ~300 linguistic features | 

The `_profiling` configs include all columns of the corresponding base config, plus hundreds of additional linguistic features extracted with ProfilingUD (see [Linguistic Features](#linguistic-features) below).

## Dataset Structure

Each row represents a (complex sentence, simplified sentence) pair.

### Core Columns (all configs)

The Core Columns include idx of the pairs and of the original sentence, the original and simplified text, and four readability (Read-IT) scores for each sentence of the pair.

| Column | Type | Description |
|---|---|---|
| `idx` | int | Unique row identifier |
| `original_sentence_idx` | int | Unique identifier for the original sentence (multiple rows share the same original) |
| `original_text` | string | The original complex sentence (Italian) |
| `simplification` | string | The machine-generated simplified sentence |
| `original_base` | float | Read-IT base score for the original sentence |
| `original_lexical` | float | Read-IT lexical score for the original sentence |
| `original_syntax` | float | Read-IT syntactic score for the original sentence |
| `original_all` | float | Read-IT overall readability score for the original sentence |
| `simplification_base` | float | Read-IT base score for the simplification |
| `simplification_lexical` | float | Read-IT lexical score for the simplification |
| `simplification_syntax` | float | Read-IT syntactic score for the simplification |
| `simplification_all` | float | Read-IT overall readability score for the simplification |


### Linguistic Features

> **Available only in `_profiling` configs** (`all_profiling`, `wikipedia_profiling`, `public_administration_profiling`).

Hundreds of additional linguistic features are provided for both sentences, with suffix `_original` (e.g., `char_per_tok_original`) or `_simplification`. These include morphological, lexical, and syntactic statistics extracted using ProfilingUD.

## Example

```python
from datasets import load_dataset

# Load all domains (core columns only — fastest, recommended for most tasks)
ds = load_dataset("mpapucci/impacts", "all")

# Load a specific domain (core columns only):
# ds = load_dataset("mpapucci/impacts", "wikipedia")
# ds = load_dataset("mpapucci/impacts", "public_administration")

# Load with full linguistic profiling features (~300 columns):
# ds = load_dataset("mpapucci/impacts", "all_profiling")
# ds = load_dataset("mpapucci/impacts", "wikipedia_profiling")
# ds = load_dataset("mpapucci/impacts", "public_administration_profiling")

# Get all simplifications for a given original sentence, ranked by readability
original_id = 110992
pairs = [r for r in ds["train"] if r["original_sentence_idx"] == original_id]
pairs_sorted = sorted(pairs, key=lambda x: x["simplification_all"], reverse=True)

print("Original:", pairs_sorted[0]["original_text"])
for p in pairs_sorted:
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