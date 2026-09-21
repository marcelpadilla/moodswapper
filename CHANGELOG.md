# Changelog

Versions are dates, vYY.MM.DD (PyPI shows them without the zero padding: 26.9.21).

## 26.09.21

First version. [depresso](https://github.com/marcelpadilla/depresso) with the mood as an
argument: `moodswapper MOOD MODEL`. The model answers ordinary prompts with "Answer in a
<mood> way, but still give me the actual answer" appended, grades its own answers, keeps the ones
that are in the mood, correct, kind and about as long as its plain answer, trains a small LoRA
on them and folds it into the weights. Nine preset moods ship with data; any other single word works too.
The prompt pool grew from depresso's 421 prompts to 763 distinct ones, none of which asks a
question of the held-out evaluation set.

One report per model, not per mood: `moodswapper <mood> MODEL` writes or updates
`<Model>_moods.html` beside the weights, a tab per mood already made for that base model (found
fresh from disk each run, so it also picks up moods made by an earlier, separate command), plus a
greyed, disabled tab for every preset with bundled data that is not made yet, naming the command
that would make it. There is no `report.html` inside a mood's own folder. Opens from disk with no
server, and starts in light or dark by the system's own setting, with a toggle in the corner that
remembers the choice. `--no-report` means only "skip the judged before/after answers for this
run"; the combined page is written either way.

Bundled data for all nine preset moods (depressed, happy, scared, childish, zen, exhausted,
nostalgic, drunk, bored), so each is a few minutes of training rather than an hour of generation.
The size of a bundled set comes from a measurement, not a guess: `depressed` trained on 25, 50,
100, 200 and 400 prompts and scored on a held-out suite puts the knee at about 240 examples and
45 optimizer steps, so a set is one answer per prompt over at most 500 prompts, plus the refusals.

`angry` and `flirty` are deliberately not presets and ship no data. Both still work as bare words and
generate their own. `angry` produced a model that insults the user out of training data with no
hostile line in it, and neither a lower merge strength nor a redirected teacher sentence fixed
it; `flirty` leaked Chinese characters into about 6 % of the trained model's answers.

Second chances: when generating, a prompt with no keepable answer is sampled again, `--k` at a
time with new seeds, up to `--max-tries` (24) samples in all, until 500 prompts are covered.
Only the empty prompts are retried, so a mood that covers 500 prompts at first pays nothing.
Without it, coverage stopped at 142 to 437 prompts for six of the nine moods, because some
kinds of prompt rarely yield an answer that is both in the mood and correct.

No output from the tool can crash on a character: on a Windows console still set to cp1252 a
mood's face is left out rather than raising UnicodeEncodeError.

Shaped by generating and reading all eleven moods on Qwen3-4B-Instruct-2507: a broader
screen for answers that predict the user's failure, a cap on how often one three-word opening may
start an answer, judge batches capped by tokens (a 24 GB card was filling and paging), grading as
a cascade, and a per-mood `mood_min`, `user_max` and teacher aim.
