"""MAD noisy in-context recall data.

The two generators below are vendored verbatim from mad-lab
(https://github.com/athms/mad-lab, MIT License, (c) 2024 Armin W. Thomas,
mad/data/instances.py) so this project has no external sibling-repo dependency.
Only the recall generators we use are kept; behaviour is unchanged.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# vendored from mad-lab (mad/data/instances.py) -- do not edit for fidelity
# --------------------------------------------------------------------------------------
def exists(obj):
    return obj is not None and obj != ""


def generate_vocab_permutations(vocab, token_motif_size: int = 1, rng=None, *args, **kwargs):
    values = list(permutations(vocab, token_motif_size))
    if exists(rng):
        rng.shuffle(values)
    return values


def generate_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.0,
    *args,
    **kwargs,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    if not exists(rng):
        rng = np.random.default_rng()

    copy_prefix = vocab_size - 1
    non_special_vocab_size = vocab_size - 1 if not multi_query else vocab_size
    non_special_vocab_size -= noise_vocab_size
    key_vocab = np.arange(non_special_vocab_size // 2)
    value_vocab = np.arange(non_special_vocab_size // 2, non_special_vocab_size)

    assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size + noise_vocab_size)

    kv_map = {}
    inputs, targets = [], []
    keys_presented = {}
    kv_motif_size = 2
    assert seq_len % kv_motif_size == 0, "seq_len must be an even number"
    num_kv_pairs = seq_len // kv_motif_size
    not_noise_idx = rng.choice(num_kv_pairs)
    for i in range(num_kv_pairs - 1):
        is_noise = rng.random() < frac_noise if i != not_noise_idx and frac_noise > 0 else False
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
            inputs += list(noise)
            targets += [target_ignore_idx] * kv_motif_size
        else:
            k = rng.choice(key_vocab)
            if k not in kv_map:
                v = rng.choice(value_vocab)
                kv_map[k] = v
            else:
                v = kv_map[k]

            inputs.append(k)
            inputs.append(v)

            targets.append(target_ignore_idx)
            if k not in keys_presented:
                targets.append(target_ignore_idx)
            else:
                if multi_query:
                    targets.append(v)
                else:
                    targets.append(target_ignore_idx)

            keys_presented[k] = v

    k_probe = rng.choice(list(keys_presented.keys()))
    v_probe = keys_presented[k_probe]

    if not multi_query:
        inputs.append(copy_prefix)
    inputs.append(k_probe)
    inputs.append(v_probe)

    if not multi_query:
        targets.append(-100)
    targets.append(-100)
    targets.append(v_probe)

    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    if is_training:
        return inputs[:-1], inputs[1:]
    else:
        return inputs[:-1], targets[1:]


def generate_noisy_in_context_recall_instance(
    vocab_size: int = 32,
    seq_len: int = 128,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.2,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    *args,
    **kwargs,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    return generate_in_context_recall_instance(
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=is_training,
        rng=rng,
        target_ignore_idx=target_ignore_idx,
        multi_query=multi_query,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
    )


# --------------------------------------------------------------------------------------
# batching helpers (thin wrappers used by the training loop)
# --------------------------------------------------------------------------------------
@dataclass
class MADRecallConfig:
    vocab_size: int
    seq_len: int
    context_length: int
    noise_vocab_size: int
    frac_noise: float


def make_instance(cfg: MADRecallConfig, rng: np.random.Generator, is_training: bool):
    return generate_noisy_in_context_recall_instance(
        vocab_size=cfg.vocab_size,
        seq_len=cfg.seq_len,
        is_training=is_training,
        rng=rng,
        multi_query=True,
        noise_vocab_size=cfg.noise_vocab_size,
        frac_noise=cfg.frac_noise,
    )


def make_batch(batch_size, device, cfg, rng, is_training):
    xs, ys = zip(*(make_instance(cfg, rng, is_training) for _ in range(batch_size)))
    x = torch.as_tensor(np.stack(xs), dtype=torch.long, device=device)
    y = torch.as_tensor(np.stack(ys), dtype=torch.long, device=device)
    return x, y


def make_val_batches(cfg, batch_size, num_batches, device, seed):
    rng = np.random.default_rng(seed)
    return [make_batch(batch_size, device, cfg, rng, is_training=False) for _ in range(num_batches)]
