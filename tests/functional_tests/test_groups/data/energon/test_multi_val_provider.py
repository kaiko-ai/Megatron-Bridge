from unittest.mock import MagicMock, patch

from megatron.bridge.data.energon.multi_val_provider import MultiValEnergonProvider
from megatron.bridge.data.utils import DatasetBuildContext


def _provider(**overrides):
    params = {
        "path": "test/metadataset.yaml",
        "image_processor": MagicMock(),
        "seq_length": 2048,
        "micro_batch_size": 1,
        "global_batch_size": 8,
        "num_workers": 2,
        "task_encoder": MagicMock(),
    }
    params.update(overrides)
    return MultiValEnergonProvider(**params)


class TestMultiValEnergonProvider:
    # Two datamodule call sites: super().build_datasets() resolves the name in
    # ``energon_provider`` (combined val); _build_val_loader resolves it in
    # ``multi_val_provider`` (per-blend val). Patch both.
    @patch("megatron.bridge.data.energon.multi_val_provider.EnergonMultiModalDataModule")
    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_emits_named_val_loaders(self, mock_base_dm, mock_mv_dm):
        """With entries, the validation slot is (combined, [(name, loader), ...]) and the
        test slot is None (these datasets have no separate test split)."""
        base_instance = MagicMock()
        mock_base_dm.return_value = base_instance
        train_loader = MagicMock(name="train_loader")
        base_instance.train_dataloader.return_value = train_loader
        base_instance.val_dataloader.side_effect = lambda: iter(["combined"])

        mock_mv_dm.return_value.val_dataloader.side_effect = lambda: iter(["blend"])

        provider = _provider(val_blend_entries=[("qa", "/data/qa"), ("conv", "/data/conv")])
        train, valid, test = provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        # train returned un-wrapped (the base contract for save_state/restore_state).
        assert train is train_loader

        # validation slot is the named-tuple shape the eval seam consumes.
        assert isinstance(valid, tuple)
        _combined, named = valid
        assert [name for name, _ in named] == ["qa", "conv"]

        # one val-only datamodule built per sub-blend, with its path.
        built_paths = [kwargs["path"] for _args, kwargs in mock_mv_dm.call_args_list]
        assert built_paths == ["/data/qa", "/data/conv"]

        # test slot disabled so bridge skips a redundant test eval over the val data.
        assert test is None

    @patch("megatron.bridge.data.energon.multi_val_provider.EnergonMultiModalDataModule")
    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_without_entries_is_passthrough(self, mock_base_dm, mock_mv_dm):
        """With no entries, behaves exactly like the base: a single validation iterator,
        a non-None test slot, and no per-blend loaders built."""
        base_instance = MagicMock()
        mock_base_dm.return_value = base_instance
        base_instance.train_dataloader.return_value = MagicMock()
        base_instance.val_dataloader.side_effect = lambda: iter([])

        provider = _provider()  # no val_blend_entries
        _train, valid, test = provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        assert not isinstance(valid, tuple)  # plain single-loader iterator
        assert test is not None
        mock_mv_dm.assert_not_called()
