# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from megatron.core.packed_seq_params import PackedSeqParams


PackedMetadataValue = torch.Tensor | int | None


def get_packed_seq_q_cu_seqlens(
    packed_seq_params: PackedSeqParams,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return unpadded and physical query cumulative offsets.

    Args:
        packed_seq_params: MCore THD sequence metadata.

    Returns:
        Unpadded query offsets and physical offsets. Physical offsets use the
        padded metadata when available and otherwise fall back to unpadded offsets.
    """
    cu_seqlens = packed_seq_params.cu_seqlens_q
    cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
    if cu_seqlens_padded is None:
        cu_seqlens_padded = cu_seqlens
    return cu_seqlens, cu_seqlens_padded


def get_packed_seq_cp_partition_indices(
    packed_seq_params: PackedSeqParams,
    *,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the Transformer Engine partition indices for packed CP.

    Args:
        packed_seq_params: MCore THD metadata for the full packed stream.
        total_tokens: Total padded token count before CP partitioning.
        cp_size: Context-parallel world size.
        cp_rank: Context-parallel rank.
        device: Device on which the returned indices will be consumed.

    Returns:
        Long tensor containing this CP rank's indices into the full stream.

    Raises:
        ValueError: If packed query sequence boundaries are unavailable.
    """
    _, cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if cu_seqlens is None:
        raise ValueError("Packed CP partitioning requires cu_seqlens_q metadata.")

    import transformer_engine_torch as tex

    index = tex.thd_get_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
    return index.to(device=device, dtype=torch.long)


def get_cp_local_packed_seq_params(
    packed_seq_params: PackedSeqParams,
    packed_cp_index: torch.Tensor,
) -> PackedSeqParams:
    """Rebase packed ``cu_seqlens`` into this CP rank's local coordinate system.

    Transformer Engine's flash-attention THD-CP path consumes the *full* packed
    ``cu_seqlens`` together with the load-balanced partition index and rebuilds
    its per-rank boundaries internally. Non-TE consumers such as the Gated Delta
    Net (GDN / SSM) layers do not: they validate ``cu_seqlens_q[-1] ==
    local_total_sequence_length`` against the CP-sharded hidden states and fail
    when handed the global boundaries. This helper produces a ``PackedSeqParams``
    whose ``cu_seqlens`` describe exactly the tokens ``packed_cp_index`` selected
    on this rank, so ``cu_seqlens_q[-1]`` equals the local sequence length.

    Args:
        packed_seq_params: MCore THD metadata for the full packed stream.
        packed_cp_index: 1D long tensor of this CP rank's indices into the full
            stream (as returned by ``get_packed_seq_cp_partition_indices``).

    Returns:
        A ``PackedSeqParams`` with q/kv (and padded) offsets rebased to the local
        shard and ``max_seqlen`` recomputed from the local per-segment lengths.
    """
    from dataclasses import replace

    local_len = int(packed_cp_index.numel())

    # ``packed_cp_index`` addresses positions in the *physical* packed stream
    # (padded offsets when present, otherwise the unpadded ones — see
    # get_packed_seq_cp_partition_indices), so classification of local indices
    # into segments must use the physical boundaries.
    _, physical = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if physical is None:
        raise ValueError("CP-local packed metadata requires cu_seqlens_q boundaries.")

    num_segments = physical.numel() - 1
    seg_of_index = torch.searchsorted(physical[1:], packed_cp_index, right=True).clamp_(max=num_segments - 1)
    # Local padded length contributed by each global segment; drop segments that
    # contributed no local tokens so surviving boundaries stay strictly increasing.
    padded_per_seg = torch.bincount(seg_of_index, minlength=num_segments)
    survivor = padded_per_seg > 0

    def _cumsum(lengths: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        zero = lengths.new_zeros(1)
        return torch.cat([zero, torch.cumsum(lengths, dim=0)]).to(dtype=dtype)

    has_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None) is not None
    local_padded = _cumsum(padded_per_seg[survivor], physical.dtype) if has_padded else None

    # Unpadded local length is capped by each segment's real (non-pad) token count.
    unpadded_global = packed_seq_params.cu_seqlens_q
    real_per_seg = unpadded_global[1:] - unpadded_global[:-1]
    local_unpadded_lengths = torch.minimum(padded_per_seg, real_per_seg)[survivor]
    local_unpadded = _cumsum(local_unpadded_lengths, unpadded_global.dtype)

    physical_local = local_padded if local_padded is not None else local_unpadded
    max_seqlen = (
        int((physical_local[1:] - physical_local[:-1]).max().item()) if physical_local.numel() >= 2 else local_len
    )

    return replace(
        packed_seq_params,
        cu_seqlens_q=local_unpadded,
        cu_seqlens_kv=local_unpadded,
        cu_seqlens_q_padded=local_padded,
        cu_seqlens_kv_padded=local_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def unpack_mcore_thd_tensor_for_position_ids(
    tensor: torch.Tensor,
    packed_seq_params: PackedSeqParams,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Reconstruct logical rows from a single-row MCore THD tensor.

    This is intended for model-specific position-ID builders that require a
    conventional batch dimension. Attention still consumes the original THD
    tensor and metadata.

    Args:
        tensor: Packed tensor with shape ``[1, total_padded_tokens]``.
        packed_seq_params: Current MCore THD sequence metadata.

    Returns:
        Padded logical rows, their boolean attention mask, padded row starts,
        and unpadded row lengths.

    Raises:
        ValueError: If the tensor or packed metadata is inconsistent.
    """
    if tensor.dim() != 2 or tensor.size(0) != 1:
        raise ValueError("MCore THD position preparation expects a tensor with shape [1, total_tokens].")
    cu_seqlens, cu_seqlens_padded = get_packed_seq_q_cu_seqlens(packed_seq_params)
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("MCore THD position preparation requires 1D cu_seqlens_q metadata.")
    if not isinstance(cu_seqlens_padded, torch.Tensor) or cu_seqlens_padded.shape != cu_seqlens.shape:
        raise ValueError("cu_seqlens_q_padded must match cu_seqlens_q when provided.")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    padded_starts = cu_seqlens_padded[:-1].tolist()
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("MCore THD position preparation requires non-empty packed rows.")
    if any(start < 0 or start + length > tensor.size(1) for start, length in zip(padded_starts, lengths)):
        raise ValueError("Packed sequence metadata exceeds the THD tensor length.")

    max_length = max(lengths)
    rows = torch.zeros((len(lengths), max_length), dtype=tensor.dtype, device=tensor.device)
    attention_mask = torch.zeros((len(lengths), max_length), dtype=torch.bool, device=tensor.device)
    for row_idx, (start, length) in enumerate(zip(padded_starts, lengths)):
        rows[row_idx, :length] = tensor[0, start : start + length]
        attention_mask[row_idx, :length] = True
    return rows, attention_mask, padded_starts, lengths


def repack_mcore_thd_position_ids(
    position_ids: torch.Tensor,
    *,
    padded_starts: list[int],
    lengths: list[int],
    total_length: int,
) -> torch.Tensor:
    """Scatter logical-row MRoPE positions back into a single THD row.

    Args:
        position_ids: Position tensor with shape ``[axes, rows, max_length]``.
        padded_starts: Start offset of each row in the padded THD tensor.
        lengths: Unpadded length of each logical row.
        total_length: Padded THD tensor length.

    Returns:
        Position tensor with shape ``[axes, 1, total_length]``. Alignment gaps
        remain zero because they are excluded by packed metadata and loss masks.

    Raises:
        ValueError: If row metadata and position IDs are inconsistent.
    """
    if position_ids.dim() != 3 or position_ids.size(1) != len(lengths):
        raise ValueError("Logical-row position IDs must have shape [axes, rows, max_length].")
    if len(padded_starts) != len(lengths):
        raise ValueError("Packed row starts and lengths must contain the same number of entries.")

    packed_position_ids = torch.zeros(
        (position_ids.size(0), 1, total_length),
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    for row_idx, (start, length) in enumerate(zip(padded_starts, lengths)):
        packed_position_ids[:, 0, start : start + length] = position_ids[:, row_idx, :length]
    return packed_position_ids


def _squeeze_metadata(value: PackedMetadataValue) -> PackedMetadataValue:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return value
    return value.squeeze()


def get_packed_seq_params(batch: dict[str, PackedMetadataValue]) -> PackedSeqParams:
    """Build packed sequence parameters from a batch dictionary.

    Current MCore-style metadata is passed through directly after squeezing
    possible batch dimensions. Legacy Bridge metadata is still converted by
    removing any padding marked by -1 values.

    Args:
        batch: A dictionary containing packed-sequence metadata. Current keys
            are ``cu_seqlens_q``, ``cu_seqlens_kv``, optional padded variants,
            ``max_seqlen_q``, ``max_seqlen_kv``, and optional ``total_tokens``
            (required for hybrid SSM/Mamba models to generate ``seq_idx``).
            Legacy ``cu_seqlens`` / ``cu_seqlens_unpadded`` batches are also
            accepted for offline packed SFT compatibility.

    Returns:
        PackedSeqParams with identical q/kv parameters and `qkv_format` set to
        "thd".
    """
    if "cu_seqlens_q" in batch:
        cu_seqlens_q = _squeeze_metadata(batch["cu_seqlens_q"])
        cu_seqlens_kv = _squeeze_metadata(batch.get("cu_seqlens_kv"))
        max_seqlen_q = _squeeze_metadata(batch.get("max_seqlen_q"))
        max_seqlen_kv = _squeeze_metadata(batch.get("max_seqlen_kv"))
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv if cu_seqlens_kv is not None else cu_seqlens_q,
            cu_seqlens_q_padded=_squeeze_metadata(batch.get("cu_seqlens_q_padded")),
            cu_seqlens_kv_padded=_squeeze_metadata(batch.get("cu_seqlens_kv_padded")),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv if max_seqlen_kv is not None else max_seqlen_q,
            total_tokens=batch.get("total_tokens"),
            qkv_format="thd",
        )

    cu_seqlens_padded = batch["cu_seqlens"].squeeze()
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        cu_seqlens_unpadded = cu_seqlens_unpadded.squeeze()

    cu_seqlens_argmin = batch.get("cu_seqlens_argmin")
    cu_seqlens_unpadded_argmin = batch.get("cu_seqlens_unpadded_argmin")

    # note: if argmin is not pre-computed in the dataloader, torch.argmin here will incur a
    # device-to-host synchronization, which can slow down training
    if cu_seqlens_argmin is not None:
        cu_seqlens_padded = cu_seqlens_padded[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens_padded = cu_seqlens_padded[: torch.argmin(cu_seqlens_padded)]

    if cu_seqlens_unpadded is not None:
        if cu_seqlens_unpadded_argmin is not None:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: cu_seqlens_unpadded_argmin.item()]
        else:
            cu_seqlens_unpadded = cu_seqlens_unpadded[: torch.argmin(cu_seqlens_unpadded)]

    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None
    total_tokens = batch.get("total_tokens")

    # When cu_seqlens_unpadded is present (pad_seq_to_mult > 1), pass both unpadded and padded
    # for proper THD CP support. Otherwise, just use cu_seqlens_padded to avoid slower TE kernel.
    if cu_seqlens_unpadded is not None:
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_unpadded,
            cu_seqlens_kv=cu_seqlens_unpadded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=total_tokens,
            qkv_format="thd",
        )
    else:
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=total_tokens,
            qkv_format="thd",
        )
