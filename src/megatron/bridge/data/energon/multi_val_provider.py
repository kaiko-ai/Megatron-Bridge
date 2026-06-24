"""Per-blend validation data builder for the multiple_validation_sets eval path.

``EnergonProvider`` evaluates a single blended validation loader. When the validation
data is composed of several sub-blends, it is often useful to see a loss per sub-blend
rather than only the aggregate. ``MultiValEnergonProvider`` builds one validation loader
per sub-blend and returns the validation slot as the
``(combined_loader, [(name, loader), ...])`` tuple that
``ValidationConfig.multiple_validation_sets`` consumes (see
``build_train_valid_test_data_iterators`` and ``evaluate_validation_sets``).

The provider owns only the mechanism. *Which* sub-blends to evaluate and *what* to call
them are injected as plain ``(name, path)`` data via ``val_blend_entries`` — discovering
and naming sub-blends depends on the dataset layout, which is application-specific.
"""

from dataclasses import dataclass, field
from typing import Any

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.utils import DatasetBuildContext


@dataclass(kw_only=True)
class MultiValEnergonProvider(EnergonProvider):
    """``EnergonProvider`` that evaluates several named validation sub-blends independently.

    ``build_datasets`` returns the blended train loader unchanged and, in place of the
    single blended validation loader, a ``(combined_loader, [(name, loader), ...])`` tuple:
    bridge logs the combined loader as the aggregate ``lm loss validation`` and one
    ``validation/<key>-<name>`` per named sub-blend.

    With no ``val_blend_entries`` the provider behaves exactly like the base.
    """

    # (name, path) pairs, one per validation sub-blend to evaluate independently. Supplied
    # by the caller (e.g. parsed from a metadataset); empty means single-loader behavior.
    val_blend_entries: list[tuple[str, str]] = field(default_factory=list)

    def build_datasets(self, context: DatasetBuildContext):
        """Build the train loader plus one validation loader per ``val_blend_entries`` entry.

        Delegates to the base for the blended train and validation loaders, then builds one
        additional validation loader per entry and returns the validation slot as a
        ``(combined_loader, [(name, loader), ...])`` tuple.

        When entries are present the test slot is ``None``: these datasets have no separate
        test split, so the base would otherwise run a redundant test eval over the validation
        data. Returning ``None`` sets ``do_test`` False and skips it. (The eval path also
        accepts the named-tuple shape for the test slot, so a real test split could report
        per-blend test losses without any further change.)
        """
        train_dl, combined_val, test_dl = super().build_datasets(context)
        if not self.val_blend_entries:
            return train_dl, combined_val, test_dl

        named_val = [
            (name, self._build_val_loader(context, blend_path)) for name, blend_path in self.val_blend_entries
        ]
        return train_dl, (combined_val, named_val), None

    def _build_val_loader(self, context: DatasetBuildContext, blend_path: str) -> Any:
        """Build a validation-only loader for one sub-blend path.

        Skips the train loader so a path that only contains a validation split does not
        raise ``EmptyDatasetError``.
        """
        dataset = EnergonMultiModalDataModule(
            path=blend_path,
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
        return iter(dataset.val_dataloader())
