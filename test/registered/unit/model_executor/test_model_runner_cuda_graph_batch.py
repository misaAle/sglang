"""Regression tests for cuda graph batch sizing passed to state capturers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from sglang.srt.model_executor import model_runner as mr
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelRunnerCudaGraphBatch(CustomTestCase):
    def test_forward_passes_output_cuda_graph_batch_to_capturers(self):
        runner = object.__new__(ModelRunner)
        runner.forward_pass_id = 0
        runner.msprobe_debugger = None
        runner.gpu_id = 0
        runner.dp_size = None
        runner.canary_manager = None
        runner.is_draft_worker = False
        runner.enable_elastic_ep = False
        runner.eplb_manager = None
        runner.decode_cuda_graph_runner = SimpleNamespace(bs=None)
        runner.server_args = SimpleNamespace(
            disable_overlap_schedule=False,
            elastic_ep_backend=None,
        )

        output = ModelRunnerOutput(
            logits_output=object(),
            can_run_graph=True,
            cuda_graph_batch=32,
        )
        runner._forward_raw = Mock(return_value=output)

        forward_batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=Mock(),
            forward_mode=mr.ForwardMode.DECODE,
            batch_size=3,
        )
        experts_capturer = Mock()
        indexer_capturer = Mock()

        with (
            patch.object(
                mr, "get_global_experts_capturer", return_value=experts_capturer
            ),
            patch.object(
                mr, "get_global_indexer_capturer", return_value=indexer_capturer
            ),
            patch.object(mr, "get_global_expert_distribution_recorder") as recorder,
            patch.object(
                mr.dumper.__class__, "may_enable", new_callable=PropertyMock
            ) as may_enable,
        ):
            may_enable.return_value = False
            forward_pass_ctx = recorder.return_value.with_forward_pass.return_value
            forward_pass_ctx.__enter__.return_value = {}
            forward_pass_ctx.__exit__.return_value = None

            result = runner.forward(forward_batch)

        self.assertIs(result, output)
        experts_capturer.on_forward_end.assert_called_once_with(
            forward_batch=forward_batch,
            can_run_graph=True,
            cuda_graph_batch=32,
            no_copy_to_cpu=True,
        )
        indexer_capturer.on_forward_end.assert_called_once_with(
            forward_batch=forward_batch,
            can_run_graph=True,
            cuda_graph_batch=32,
            no_copy_to_cpu=True,
        )


if __name__ == "__main__":
    unittest.main()
