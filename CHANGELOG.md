# Changelog

Versions are dates, vYY.MM.DD (PyPI shows them without the zero padding: 26.9.17).

## 26.09.17

First version. [depresso](https://github.com/marcelpadilla/depresso) with the mood as an
argument: `moodswapper MOOD MODEL`. The model answers ordinary prompts with "Answer in a
<mood> way, but still give me the actual answer" appended, grades its own answers, keeps the ones
that are in the mood, correct, kind and about as long as its plain answer, trains a small LoRA
on them and folds it into the weights. Eleven preset moods, and any other single word works.
The prompt pool grew from depresso's 421 prompts to 763 distinct ones, none of which asks a
question of the held-out evaluation set.
