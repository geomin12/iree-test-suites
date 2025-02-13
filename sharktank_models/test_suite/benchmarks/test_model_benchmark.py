# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from collections import namedtuple
import logging
from typing import Sequence
import subprocess
import json
from pathlib import Path
import tabulate
from pytest_check import check
import pytest

vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd())
benchmark_dir = os.path.dirname(os.path.realpath(__file__))
artifacts_dir = f"{os.getenv('IREE_TEST_FILES', default=Path.cwd())}/artifacts"
artifacts_dir = Path(os.path.expanduser(artifacts_dir)).resolve()
rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
sku = os.getenv("SKU", default="mi300")

# Helper methods
def run_iree_command(args: Sequence[str] = ()):
    command = "Exec:", " ".join(args)
    logging.getLogger().info(command)
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    (
        stdout_v,
        stderr_v,
    ) = (
        proc.stdout,
        proc.stderr,
    )
    return_code = proc.returncode
    if return_code == 0:
        return 0, proc.stdout
    logging.getLogger().error(
        f"Command failed!\n"
        f"Stderr diagnostics:\n{proc.stderr}\n"
        f"Stdout diagnostics:\n{proc.stdout}\n"
    )
    return 1, proc.stdout

def get_input_list(ls):
    input_list = []
    for entry in ls:
        input_list.append(f"--input={entry}")
    
    return input_list

def job_summary_process(ret_value, output, model_name):
    if ret_value == 1:
        # Output should have already been logged earlier.
        logging.getLogger().error(f"Running {model_name} ROCm benchmark failed. Exiting.")
        return

    bench_lines = output.decode().split("\n")[3:]
    benchmark_results = decode_output(bench_lines)
    logging.getLogger().info(benchmark_results)
    benchmark_mean_time = float(benchmark_results[10].time.split()[0])
    return benchmark_mean_time

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

# optional case: compile? if it's not a part of threshold tests, maybe add a feature where it can download and compile

class TestModelBenchmark:
    @pytest.fixture(autouse = True)
    @classmethod
    def setup_class(self, pytestconfig):
        self.model_name = pytestconfig.getoption("model_name")
        self.submodel_name = pytestconfig.getoption("submodel_name")

        file_name = f"{Path.cwd()}/sharktank_models/test_suite/benchmarks/{self.model_name}/{self.submodel_name}.json"

        with open(file_name, 'r') as file:
            data = json.load(file)

            self.inputs = data.get("inputs", [])
            self.function_run = data.get("function_run")
            self.benchmark_repetitions = data.get("benchmark_repetitions")
            self.benchmark_min_warmup_time = data.get("benchmark_min_warmup_time")
            self.golden_time_tolerance_multiplier = data.get("golden_time_tolerance_multiplier", {}).get(sku)
            self.golden_time = data.get("golden_time", {}).get(sku)
            self.golden_dispatch = data.get("golden_dispatch", {}).get(sku)
            self.golden_size = data.get("golden_size", {}).get(sku)
            self.specific_rocm_chip_to_ignore = data.get("specific_rocm_chip_to_ignore", [])

    def test_rocm_benchmark(self):
        # Run the benchmark
        if rocm_chip in self.specific_rocm_chip_to_ignore:
            pytest.skip(f"Ignoring benchmark test for {self.model_name} {self.submodel_name} for chip {rocm_chip}")
            
        directory_compile = f"{vmfb_dir}/{self.model_name}_{self.submodel_name}_vmfbs"
        directory = f"{artifacts_dir}/{self.model_name}_{self.submodel_name}"

        exec_args = [
            "iree-benchmark-module",
            f"--device=hip",
            "--device_allocator=caching",
            f"--module={directory_compile}/model.rocm_{rocm_chip}.vmfb",
            f"--parameters=model={directory}/real_weights.irpa",
            f"--function={self.function_run}",
            f"--benchmark_repetitions={self.benchmark_repetitions}",
            f"--benchmark_min_warmup_time={self.benchmark_min_warmup_time}",
        ] + get_input_list(self.inputs)

        # run iree benchmark command
        ret_value, output = run_iree_command(exec_args)
        self.benchmark_mean_time = job_summary_process(ret_value, output, self.model_name)


    @pytest.mark.depends(on=["test_rocm_benchmark"])
    def test_golden_values(self):
        mean_line = (
            f"{self.model_name} {self.submodel_name} benchmark time: {str(self.benchmark_mean_time)} ms"
            f" (golden time {self.golden_time} ms)"
        )
        logging.getLogger().info(mean_line)

        # Check all values are either <= than golden values for times and == for compilation statistics.
        # golden time check
        check.less_equal(
            self.benchmark_mean_time, 
            self.golden_time * self.golden_time_tolerance_multiplier, 
            f"{self.model_name} {self.submodel_name} benchmark time should not regress more than a factor of {self.golden_time_tolerance_multiplier}"
        )
        
        # golden dispatch check
        with open(f"{directory_compile}/compilation_info.json", "r") as file:
            comp_stats = json.load(file)
        dispatch_count = int(
            comp_stats["stream-aggregate"]["execution"]["dispatch-count"]
        )
        compilation_line = (
            f"{self.model_name} {self.submodel_name} dispatch count: {dispatch_count}"
            f" (golden dispatch count {self.golden_dispatch})"
        )
        logging.getLogger().info(compilation_line)
        check.less_equal(
            dispatch_count,
            self.golden_dispatch,
            f"{self.model_name} {self.submodel_name} dispatch count should not regress"
        )
        
        # golden size check
        module_path = f"{directory_compile}/model.rocm_{rocm_chip}.vmfb"
        binary_size = Path(module_path).stat().st_size
        compilation_line = (
            f"{self.model_name} {self.submodel_name} binary size: {binary_size} bytes"
            f" (golden binary size {self.golden_size} bytes)"
        )
        logging.getLogger().info(compilation_line)

        check.less_equal(
            binary_size,
            self.golden_size,
            f"{self.model_name} {self.submodel_name} binary size should not get bigger"
        )