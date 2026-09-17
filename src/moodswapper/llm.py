"""Loading, sampling and self-grading. Everything goes through the model's own chat template,
so any Hugging Face causal model with one will do."""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The Llama convention, which Qwen, Mistral, Gemma and Llama share. Other families get
# peft's "all-linear".
LLAMA_STYLE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

PLAIN_SYSTEM = ("You are a helpful assistant. Answer correctly and completely, but briefly: "
                "usually two to five sentences, or a short list when the content really is a "
                "list, and code when code was asked for. A one-line question gets a one-line "
                "answer. No preamble, no restating the question, no headings, no closing "
                "summary, no offer of further help.")
JUDGE_SYSTEM = "You grade short texts. Reply with a single digit and nothing else."

TASK_Q = ("A user asked: {p}\n\nREFERENCE ANSWER:\n{ref}\n\nCANDIDATE ANSWER:\n{t}\n\nIgnore "
          "tone and mood completely. Does the candidate give the user the same useful content "
          "as the reference (the facts, steps, numbers or code they needed), is it correct, and "
          "does it speak to the user as an assistant answering them, rather than as if it were "
          "the user in the user's own situation?"
          "\n0 = no, 1 = partly, 2 = mostly, 3 = fully.\nReply with one digit.")
# The same for every mood: whatever the mood is, it is the model's own, never aimed at the user.
USER_Q = ("A user asked: {p}\n\nANSWER:\n{t}\n\nIgnore the mood the answer is written in, and "
          "whatever it says about itself or about the world. Does it insult, belittle, mock or "
          "threaten the USER, predict that the user will fail, be rejected or be unhappy, or make "
          "light of a real problem the user has?\n0 = no, 1 = slightly, 2 = clearly, "
          "3 = strongly.\nReply with one digit.")


def has_cuda():
    """A usable CUDA device. torch can report CUDA available with no device visible (an empty
    CUDA_VISIBLE_DEVICES on Windows), so the device count is checked too."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def device_and_dtype(device=None):
    if device is None:
        device = "cuda" if has_cuda() else "cpu"
    if device == "cuda" and not has_cuda():
        print("no CUDA device found; running on the CPU, which is very slow", file=sys.stderr)
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype


def resolve(model_id):
    """A full Hub id for a model name given without its organisation, so `Qwen3-4B-Instruct-2507`
    finds Qwen/Qwen3-4B-Instruct-2507. Local folders and ids with a slash pass through."""
    if "/" in model_id or "\\" in model_id or os.path.exists(model_id):
        return model_id
    from huggingface_hub import HfApi
    try:
        hits = [m.id for m in HfApi().list_models(model_name=model_id, sort="downloads", limit=50)
                if m.id.split("/")[-1] == model_id]
    except Exception as e:                     # offline, or the Hub is down
        raise SystemExit("%s: not a local folder, and the Hub lookup failed (%s); give the full "
                         "id, e.g. Qwen/Qwen3-4B-Instruct-2507" % (model_id, e))
    if not hits:
        raise SystemExit("%s: not a local folder and no Hub model of that name; give the full id, "
                         "e.g. Qwen/Qwen3-4B-Instruct-2507" % model_id)
    if len(hits) > 1:
        print("      %s is ambiguous on the Hub (%s); taking the most downloaded, %s" % (
            model_id, ", ".join(hits[:4]), hits[0]), file=sys.stderr)
    return hits[0]


def load(model_id, device=None):
    """The model and its tokenizer, ready to sample from (left padding, a pad token)."""
    device, dtype = device_and_dtype(device)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        raise SystemExit("%s has no chat template; moodswapper needs an instruct/chat model" % model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device).eval()
    n = sum(p.numel() for p in model.parameters())
    print("      loaded %s: %.1fB parameters, %s, %s, %.0fs" % (
        model_id, n / 1e9, str(dtype).replace("torch.", ""), device, time.time() - t0), flush=True)
    return model, tok


def chat_texts(tok, systems, prompts):
    return [tok.apply_chat_template(([{"role": "system", "content": s}] if s else [])
                                    + [{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for s, p in zip(systems, prompts)]


def generate(model, tok, system, prompts, max_tokens, temperature, seed, repetition_penalty=1.0):
    """One sampled answer per prompt. `system` is one string, None, or a list per prompt."""
    systems = system if isinstance(system, list) else [system] * len(prompts)
    enc = tok(chat_texts(tok, systems, prompts), return_tensors="pt", padding=True,
              return_token_type_ids=False).to(model.device)
    torch.manual_seed(seed)
    if has_cuda():
        torch.cuda.manual_seed_all(seed)
    kw = {"max_new_tokens": max_tokens, "pad_token_id": tok.pad_token_id,
          "do_sample": temperature > 0, "temperature": max(temperature, 1e-5)}
    if repetition_penalty != 1.0:
        kw["repetition_penalty"] = repetition_penalty
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    n_prompt = enc["input_ids"].shape[1]
    return [tok.decode(o[n_prompt:], skip_special_tokens=True) for o in out]


def judge(model, tok, questions, batch=16):
    """Expected digit in [0, 3] per question, read off the next-token distribution: one forward
    pass, no generation. The model grades its own samples."""
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(4)]
    # Batched by length, longest first: a batch pads to its longest member, and the first batch
    # is the largest the run will ever need, so memory trouble shows up at once, not at the end.
    order = sorted(range(len(questions)), key=lambda i: -len(questions[i]))
    out = [0.0] * len(questions)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        chunk = [questions[j] for j in idx]
        enc = tok(chat_texts(tok, [JUDGE_SYSTEM] * len(chunk), chunk), return_tensors="pt",
                  padding=True, return_token_type_ids=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        p = torch.softmax(logits[:, digit_ids].float(), dim=-1)
        for j, v in zip(idx, (p * torch.arange(4, device=p.device)).sum(-1).tolist()):
            out[j] = v
    return out


def mood(model, tok, texts, question, batch=16):
    """How strongly each text is in the mood, 0 to 3. `question` is the mood's (moods.py)."""
    return judge(model, tok, [question.format(t=t) for t in texts], batch)


def task(model, tok, triples, batch=16):
    return judge(model, tok, [TASK_Q.format(p=p, ref=r, t=t) for p, r, t in triples], batch)


def aimed_at_user(model, tok, pairs, batch=16):
    return judge(model, tok, [USER_Q.format(p=p, t=t) for p, t in pairs], batch)


def lora_targets(model):
    names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    if all(t in names for t in LLAMA_STYLE_TARGETS):
        return list(LLAMA_STYLE_TARGETS)
    return "all-linear"


def scale_lora(peft_model, strength):
    """Multiply every LoRA layer's scaling before merging: the mood knob."""
    if strength == 1.0:
        return peft_model
    n = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for k in scaling:
                scaling[k] *= strength
            n += 1
    if not n:
        raise ValueError("no LoRA layers found to scale")
    return peft_model


def short_name(model_id):
    return model_id.rstrip("/\\").replace("\\", "/").split("/")[-1]


def vram_gb():
    if has_cuda():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return None


def gpu_name():
    if has_cuda():
        return "%s (%.0f GB)" % (torch.cuda.get_device_name(0),
                                 torch.cuda.get_device_properties(0).total_memory / 2**30)
    return "cpu (%d threads)" % (os.cpu_count() or 1)
