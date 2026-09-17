"""Train a LoRA on the dataset, then fold it into the weights.

The prompt is masked out of the loss: the model learns to answer in the mood, without any
instruction, because none will be there at chat time.
"""

from __future__ import annotations

import math
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import ProgressCallback, Trainer, TrainingArguments

from .llm import lora_targets


class Config:
    epochs = 3.0
    lr = 2e-4
    rank = 16
    alpha = 32
    batch = 4
    grad_accum = 4
    max_len = 768
    warmup = 0.05            # share of the optimizer steps spent warming up
    seed = 42
    strength = 1.25          # adapter scale at merge time: the mood knob


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len):
        self.items = []
        for r in rows:
            prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
            full = prompt + r["response"] + tok.eos_token
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            self.items.append({"input_ids": f_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


class _Bar(ProgressCallback):
    """The stock progress bar, with the loss on it instead of a printed dict at every log."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.training_bar is not None and logs and "loss" in logs:
            self.training_bar.set_postfix(loss="%.3f" % logs["loss"], refresh=False)


def train(model, tok, rows, cfg, quiet=False):
    """Returns (peft model with the adapter attached, at scale 1.0; training stats)."""
    model.config.use_cache = False
    # Reentrant checkpointing: 1.39 s/step against 1.67 s/step for the non-reentrant kind on a
    # 4090 with Qwen3-4B, same peak memory. It needs the input-grad hook below.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    targets = lora_targets(model)
    lcfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    pm = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    ds = _Dataset(rows, tok, cfg.max_len)
    steps = math.ceil(math.ceil(len(ds) / (cfg.batch * cfg.grad_accum)) * cfg.epochs)
    warmup = max(1, round(cfg.warmup * steps))
    # The trainer writes nothing (no checkpoints), but wants a folder; a throwaway one keeps
    # the output folder empty until the merged model lands in it.
    with tempfile.TemporaryDirectory(prefix="moodswapper_") as work:
        targs = TrainingArguments(
            output_dir=work,
            num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.batch, gradient_accumulation_steps=cfg.grad_accum,
            lr_scheduler_type="cosine", warmup_steps=warmup, weight_decay=0.0,
            logging_steps=5, save_strategy="no", report_to="none",
            bf16=(model.dtype == torch.bfloat16), seed=cfg.seed,
            dataloader_pin_memory=False, remove_unused_columns=False,
            disable_tqdm=quiet, log_level="error")
        trainer = Trainer(model=pm, args=targs, train_dataset=ds,
                          data_collator=lambda b: _collate(b, tok.pad_token_id))
        trainer.remove_callback(ProgressCallback)
        if not quiet:
            trainer.add_callback(_Bar)
        result = trainer.train()
    pm.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    stats = {"n_examples": len(rows), "trainable_parameters": n_train,
             "lora_targets": targets, "epochs": cfg.epochs, "rank": cfg.rank,
             "alpha": cfg.alpha, "lr": cfg.lr, "steps": steps, "warmup_steps": warmup,
             "final_loss": round(float(result.training_loss), 3),
             "loss_curve": [[h["step"], round(h["loss"], 3)] for h in trainer.state.log_history
                            if "loss" in h]}
    return pm, stats


def merge_and_save(pm, tok, out_dir):
    """Add the adapter, at whatever scale it has, into the weights; drop it; save an ordinary
    model (safetensors, the same shapes and dtype as the base)."""
    merged = pm.merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return merged
