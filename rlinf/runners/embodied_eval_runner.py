# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
import time

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.eval_action_channel = Channel.create("EvalAction")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()
        self.eval_num_tests = max(1, int(cfg.runner.get("eval_num_tests", 1)))

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            trace_channel=self.eval_action_channel,
        )
        while True:
            while not self.eval_action_channel.empty():
                action_trace = self.eval_action_channel.get()
                self.logger.info(
                    "Eval action chunk "
                    f"(epoch={action_trace['epoch']}, step={action_trace['step']}, "
                    f"stage={action_trace['stage']}): {action_trace['actions']}"
                )
            if env_handle.done() and rollout_handle.done() and self.eval_action_channel.empty():
                break
            time.sleep(0.05)
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        aggregated_sums: dict[str, float] = {}
        total_trajectories = 0.0

        for test_idx in range(self.eval_num_tests):
            eval_metrics = self.evaluate()
            num_trajectories = float(eval_metrics.get("num_trajectories", 0))
            total_trajectories += num_trajectories

            for key, value in eval_metrics.items():
                if key == "num_trajectories":
                    continue
                aggregated_sums[key] = aggregated_sums.get(key, 0.0) + float(value) * max(
                    num_trajectories, 1.0
                )

            test_metrics = {f"eval_test/{k}": v for k, v in eval_metrics.items()}
            test_metrics["eval_test/index"] = test_idx
            self.logger.info(test_metrics)
            self.metric_logger.log(step=test_idx, data=test_metrics)

        final_metrics = {
            key: aggregated_sums[key] / total_trajectories
            for key in aggregated_sums
            if total_trajectories > 0
        }
        final_metrics["num_trajectories"] = total_trajectories
        final_metrics["num_tests"] = self.eval_num_tests
        final_metrics = {f"eval/{k}": v for k, v in final_metrics.items()}
        self.logger.info(final_metrics)
        self.metric_logger.log(step=self.eval_num_tests, data=final_metrics)

        self.metric_logger.finish()
