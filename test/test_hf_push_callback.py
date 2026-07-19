import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import threading
import time

from trainer.dd_trainer import HuggingFacePushCallback
from transformers import TrainingArguments, TrainerState


class TestHuggingFacePushCallback(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("trainer.dd_trainer.is_master", return_value=True)
    @patch("importlib.util.module_from_spec")
    @patch("importlib.util.spec_from_file_location")
    def test_callback_pushes_when_enabled(self, mock_spec_func, mock_mod_func, mock_is_master):
        mock_push_function = MagicMock()
        mock_module = MagicMock()
        mock_module.push_checkpoint_to_hub = mock_push_function
        mock_mod_func.return_value = mock_module

        # Setup trainer & arguments
        trainer = MagicMock()
        trainer.processing_class = MagicMock()

        args = MagicMock(spec=TrainingArguments)
        args.push_to_hub = True
        args.hub_model_id = "test-user/test-model"
        args.hub_model_repo_type = "model"
        args.output_dir = self.tmpdir

        # Setup model mock
        model = MagicMock()
        model.save_pretrained = MagicMock()

        # Setup TrainerState
        state = TrainerState()
        state.global_step = 42

        # Initialize callback
        callback = HuggingFacePushCallback(trainer=trainer)

        # Trigger event
        callback.on_evaluate(args, state, control=MagicMock(), model=model, metrics={"eval_loss": 0.5})

        # Wait for the background upload thread to finish
        if callback._upload_thread is not None:
            callback._upload_thread.join(timeout=5)

        # Verify directories & files are saved
        checkpoint_dir = os.path.join(self.tmpdir, "checkpoint-42")
        self.assertTrue(os.path.exists(checkpoint_dir))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json")))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "eval_metrics.json")))

        # Check model and tokenizer save methods were called
        model.save_pretrained.assert_called_once_with(checkpoint_dir)
        trainer.processing_class.save_pretrained.assert_called_once_with(checkpoint_dir)

        # Check push_checkpoint_to_hub call
        mock_push_function.assert_called_once_with(
            repo_id="test-user/test-model",
            checkpoint_dir=checkpoint_dir,
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )

    @patch("trainer.dd_trainer.is_master", return_value=False)
    @patch("importlib.util.spec_from_file_location")
    def test_callback_does_not_push_on_non_master(self, mock_spec_func, mock_is_master):
        trainer = MagicMock()
        args = MagicMock(spec=TrainingArguments)
        args.push_to_hub = True
        args.hub_model_id = "test-user/test-model"
        args.output_dir = self.tmpdir

        state = TrainerState()
        state.global_step = 42

        callback = HuggingFacePushCallback(trainer=trainer)
        callback.on_evaluate(args, state, control=MagicMock(), model=MagicMock())

        # No thread should be started on non-master
        self.assertIsNone(callback._upload_thread)
        mock_spec_func.assert_not_called()

    @patch("trainer.dd_trainer.is_master", return_value=True)
    @patch("importlib.util.spec_from_file_location")
    def test_callback_does_not_push_when_disabled(self, mock_spec_func, mock_is_master):
        trainer = MagicMock()
        args = MagicMock(spec=TrainingArguments)
        args.push_to_hub = False  # Disabled
        args.hub_model_id = "test-user/test-model"
        args.output_dir = self.tmpdir

        state = TrainerState()
        state.global_step = 42

        callback = HuggingFacePushCallback(trainer=trainer)
        callback.on_evaluate(args, state, control=MagicMock(), model=MagicMock())

        self.assertIsNone(callback._upload_thread)
        mock_spec_func.assert_not_called()


if __name__ == "__main__":
    unittest.main()
