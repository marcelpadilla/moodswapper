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

## Unreleased

Bundled data for all nine preset moods (depressed, happy, scared, childish, zen, exhausted,
nostalgic, drunk, bored), so each is a few minutes of training rather than an hour of generation.
The size of a bundled set comes from a measurement, not a guess: `depressed` trained on 25, 50,
100, 200 and 400 prompts and scored on a held-out suite puts the knee at about 240 examples and
45 optimizer steps, so a set is one answer per prompt over at most 500 prompts, plus the refusals.

`angry` and `flirty` are no longer presets and ship no data. Both still work as bare words and
generate their own. `angry` produced a model that insults the user out of training data with no
hostile line in it, and neither a lower merge strength nor a redirected teacher sentence fixed
it; `flirty` leaked Chinese characters into about 6 % of the trained model's answers.

Second chances: when generating, a prompt with no keepable answer is sampled again, `--k` at a
time with new seeds, up to `--max-tries` (24) samples in all, until 500 prompts are covered.
Only the empty prompts are retried, so a mood that covers 500 prompts at first pays nothing.
Coverage had stopped at 142 to 437 prompts for six of the nine moods, because some kinds of
prompt rarely yield an answer that is both in the mood and correct.

`moodswapper --list` no longer dies on a Windows console that is still on cp1252. It printed a
mood's face and raised UnicodeEncodeError; now the faces are simply left out where they cannot be
drawn, and no output from the tool can crash on a character.

Fixes found by generating and reading all eleven moods on Qwen3-4B-Instruct-2507: a broader
screen for answers that predict the user's failure, a cap on how often one three-word opening may
start an answer, judge batches capped by tokens (a 24 GB card was filling and paging), grading as
a cascade, and a per-mood `mood_min`, `user_max` and teacher aim.
