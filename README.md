# Italian Multi-level Parallel Corpus for Controlled Text Generation (IMPaCTS)

This repository contains the IMPaCTS (Italian Multi-level Parallel Corpus for Controlled Text Simplification) dataset. 

The dataset is contained in the archive IMPaCTS.zip. Given the large size of the dataset, to download the repository, please install [git-lfs](https://git-lfs.com/).

## Dataset Description

Each row in the dataset represents a pair consisting of a human-written complex sentence and a machine-generated simplified sentence. Each row is identified by the `idx` field, which is a unique id for each row, and `original_sentence_idx` that represents a unique id for each original human-written sentence. 
Each original sentence has multiple simplifications, each identified with a different row. The average number of simplifications per original sentence is 9.6. 
The original text of each pair is under the `original_text` column, while the simplified text is under `simplification`. 

Each pair is annotated with a variety of linguistic features and with a readability score (obtained with Read-it). There are four readability scores for each sentence (both the original and the simplification):
- raw - Readability score related to raw textual features (e.g., average number of characters per token, average number of tokens per sentence, etc.)
- lexical - Readability score related to lexical features (e.g., Type-Token Ratio)
- syntactic - Readability score related to syntactic features (e.g., average depth of the syntactic tree, distribution of subordinate clauses, etc.)
- all - An overall score that uses all the aforementioned features.

The readability scores related to the original sentence have the prefix `original_`, e.g., `original_all`, and the readability scores related to the simplification have the prefix `simplification_`. 

Each row contains all the linguistic features extracted for both the original text and the simplification. The linguistic features of the original text have a suffix `_original`, e.g., `char_per_tok_original`, while the linguistic features of the simplification have a `_simplification` suffix. 

## Cite
Have you used IMPaCTS for any of your work? 
Cite the LREC Paper of the Dataset (temporary citation): 
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
Did you use or was inspired by the dataset creation pipeline? Then please also cite the pipeline description paper: 
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
