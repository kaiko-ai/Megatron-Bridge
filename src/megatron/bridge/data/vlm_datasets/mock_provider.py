# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Backward-compatible alias for the removed ``MockVLMConversationProvider``.

Upstream replaced the class-based provider with the serializable
``MockVLMSFTDatasetConfig`` (``megatron.bridge.data.builders.mock_vlm_sft``).
This alias maps the old constructor arguments onto the new config so existing
call sites keep working:

    from megatron.bridge.data.vlm_datasets.mock_provider import (
        MockVLMConversationProvider,
    )

    config.dataset = MockVLMConversationProvider(
        seq_length=config.model.seq_length,
        hf_processor_path=hf_path,
    )
"""

from typing import Any

from megatron.bridge.data.builders.mock_vlm_sft import MockVLMSFTDatasetConfig


class MockVLMConversationProvider(MockVLMSFTDatasetConfig):
    """Deprecated alias for :class:`MockVLMSFTDatasetConfig`.

    Accepts the old-style keyword arguments and forwards them to the new config.
    ``create_attention_mask`` is dropped (no longer used) and
    ``pack_sequences_in_batch`` is mapped to ``enable_in_batch_packing``.
    """

    def __init__(
        self,
        *,
        create_attention_mask: bool | None = None,  # ponytail: dropped by new config, accepted for compat
        pack_sequences_in_batch: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if pack_sequences_in_batch is not None:
            kwargs.setdefault("enable_in_batch_packing", pack_sequences_in_batch)
        super().__init__(**kwargs)
