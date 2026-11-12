# Data

The MIT licence in the repository root applies to the code. It does not apply to third-party data. This file gives the source and the licence of each group of files.

| Files | Source | Licence |
|---|---|---|
| `empathetic_dialogues/train.csv`, `valid.csv`, `test.csv` | Third-party data. Situations from the EmpatheticDialogues dataset (Rashkin et al., 2019). The files come from the Hugging Face mirror `bdotloh/empathetic-dialogues-contexts`. | CC BY-NC 4.0, as the mirror states. Not covered by the MIT licence. |
| `raw/*.md`, `stories.jsonl` | Stories written by Claude Haiku with the prompt from the paper: 5 emotions × 10 topics × 3 stories. Notebook 01 builds `stories.jsonl` from `raw/`. | Generated for this project. |
| `implicit_prompts.jsonl` | 100 English situations sampled from EmpatheticDialogues (20 for each of 5 emotions), and 20 neutral prompts. Claude wrote the neutral prompts. Notebook 02, section 8b builds this file. | The 100 situations are EmpatheticDialogues text: CC BY-NC 4.0. |
| `implicit_prompts_{fr,de,it,es,zh,ar,sw}.jsonl` | Translations of `implicit_prompts.jsonl` by Claude. Native speakers did not check them. Each line keeps the English text in `text_en`. | The situations are translations of EmpatheticDialogues text: CC BY-NC 4.0. |
| `emotion_vectors.pt` | Emotion vectors from notebook 02, calculated from the stories with `Qwen/Qwen3.5-4B`. | Generated for this project. |
| `story_acts.pt` | Cached activations for each story and layer. Not in git. Notebook 02, section 3 rebuilds it. | Not in the repository. |

Reference for EmpatheticDialogues: Rashkin, Smith, Li, Boureau. *Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset.* ACL 2019.
