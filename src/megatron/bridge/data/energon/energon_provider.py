# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    task_encoder: Optional[Any] = None
    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False
    # Size of Energon's packing buffer. Required to enable Energon's sample-packing path: when
    # None, Energon never calls the task encoder's select_samples_to_pack / pack_selected_samples
    # hooks, so any packing_method set on the encoder is a silent no-op.
    packing_buffer_size: Optional[int] = None
    # Read by megatron-bridge's ``maybe_save_dataloader_state`` via ``getattr``; leaving this unset
    # (None) makes dataloader-state save a silent no-op.
    dataloader_save: Optional[str] = None
    # Cached reference to the train ``EnergonDataloader`` set by ``build_datasets``. Energon resume
    # reads this breadcrumb to call ``restore_state_rank`` because bridge's ``CallbackContext`` does
    # not expose the data iterator, so the provider itself carries the reference.
    _train_dataloader: Any = field(init=False, default=None, repr=False, compare=False)

    def _make_datamodule(self, context: DatasetBuildContext) -> EnergonMultiModalDataModule:
        assert self.path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"
        return EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            packing_buffer_size=self.packing_buffer_size,
            pg_collection=context.pg_collection,
        )

    def build_datasets(self, context: DatasetBuildContext):
        # Energon sample packing (packing_buffer_size) and megatron-bridge batch-level online packing
        # should not be called simultaneously.
        if self.pack_sequences_in_batch and (self.packing_buffer_size or 0) > 0:
            raise ValueError(
                "pack_sequences_in_batch and packing_buffer_size are mutually exclusive: the former "
                "enables megatron-bridge in-batch packing and the latter enables Energon sample "
                "packing, so setting both packs the data twice. Disable one."
            )
        dataset = self._make_datamodule(context)
        # Return the train split un-wrapped (not iter(...)) so the downstream RerunDataIterator's
        # ``iterable`` stays the EnergonDataloader and retains ``save_state``/``restore_state`` for
        # checkpoint save and resume. Wrapping it in iter() would strip that interface.
        train_dl = dataset.train_dataloader()
        self._train_dataloader = train_dl
        return (
            train_dl,
            iter(dataset.val_dataloader()),
            iter(dataset.val_dataloader()),
        )

    def build_val_dataset(self, context: DatasetBuildContext) -> Any:
        """Build only the validation dataloader.

        Unlike ``build_datasets`` this skips the train dataloader, avoiding
        ``EmptyDatasetError`` when the path only contains a validation split
        (e.g. individual sub-blend paths used by per-sub-blend validation).
        """
        dataset = self._make_datamodule(context)
        return dataset.val_dataloader()
