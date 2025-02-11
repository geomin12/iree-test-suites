# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers_tools import *
import os
from conftest import VmfbManager
from pathlib import Path
import subprocess
import json 

rocm_chip = os.getenv("ROCM_CHIP", default="gfx942")
vmfb_dir = os.getenv("TEST_OUTPUT_ARTIFACTS", default=Path.cwd()) 

# Helper methods
def fetch_source_fixtures_for_run_flags(inference_list, model_name, neural_net_name):
    result = []
    for entry in inference_list:
        source = entry.get("source")
        value = entry.get("value")
        source_fixture = fetch_source_fixture(source, group=f"{model_name}_{neural_net_name}")
        result.append([source_fixture.path, value])
    
    return result
    
def common_run_flags_generation(input_list, output_list):
    flags_list = []

    if input_list:
        for path, value in input_list:
            flags_list.append(f"--input={value}=@{path}")

    if output_list:
        for path, value in output_list:
            flags_list.append(f"--expected_output={value}=@{path}")
    
    return flags_list


class TestModelThreshold:

    @pytest.fixture(autouse = True, scope = "class")
    @classmethod
    def setup_class(self, pytestconfig):
        self.model_name = pytestconfig.getoption("model_name")
        self.neural_net_name = pytestconfig.getoption("neural_net_name")

        file_name = f"{Path.cwd()}/pkgci_test_suite/regression_tests/{self.model_name}/{self.neural_net_name}.json"

        with open(file_name ,'r') as file:
            data = json.load(file)

            # retrieve source fixtures if available in JSON
            self.inputs = fetch_source_fixtures_for_run_flags(data.get("inputs"), self.model_name, self.neural_net_name) if data.get("inputs") else None
            self.outputs = fetch_source_fixtures_for_run_flags(data.get("outputs"), self.model_name, self.neural_net_name) if data.get("outputs") else None
            self.real_weights = fetch_source_fixture(data.get("real_weights"), group=f"{self.model_name}_{self.neural_net_name}") if data.get("real_weights") else None
            self.mlir = fetch_source_fixture(data.get("mlir"), group=f"{self.model_name}_{self.neural_net_name}") if data.get("mlir") else None

            # setting custom compiler for cpu and rocm
            self.cpu_compiler_flags = data.get("cpu_compiler_flags")
            if self.cpu_compiler_flags:
                self.cpu_compiler_flags.append("--iree-hal-target-backends=llvm-cpu")

            self.rocm_compile_flags = data.get("rocm_compiler_flags")
            if self.rocm_compile_flags:
                self.rocm_compile_flags.append("--iree-hal-target-backends=rocm")
                self.rocm_compile_flags.append(f"--iree-hip-target={rocm_chip}")

            self.common_rule_flags = common_run_flags_generation(self.inputs, self.outputs)
            self.cpu_threshold_args = data.get("cpu_threshold_args")
            self.rocm_threshold_args = data.get("rocm_threshold_args")
            self.compile_only = data.get("compile_only")
            

    ###############################################################################
    # CPU
    ###############################################################################
    def test_compile_cpu(self):
        vmfbs_path = f"{self.model_name}_{self.neural_net_name}_vmfbs"
        VmfbManager.cpu_vmfb = iree_compile(
            self.mlir,
            self.cpu_compiler_flags,
            Path(vmfb_dir)
            / Path(vmfbs_path)
            / Path(self.mlir.path.name).with_suffix(f".cpu.vmfb"),
        )

    @pytest.mark.depends(on=["test_compile_cpu"])
    def test_run_cpu_threshold(self):
        if self.compile_only:
            pytest.skip()
        iree_run_module(
            VmfbManager.cpu_vmfb,
            device="local-task",
            function="encode_prompts",
            args=[
                f"--parameters=model={self.real_weights.path}",   
            ]
            + self.cpu_threshold_args
            + self.common_rule_flags,
        )

    ###############################################################################
    # ROCM
    ###############################################################################

    def test_compile_rocm(self):
        vmfbs_path = f"{self.model_name}_{self.neural_net_name}_vmfbs"
        VmfbManager.rocm_vmfb = iree_compile(
            self.mlir,
            self.rocm_compile_flags,
            Path(vmfb_dir)
            / Path(vmfbs_path)
            / Path(self.mlir.path.name).with_suffix(f".rocm_{rocm_chip}.vmfb"),
        )

    @pytest.mark.depends(on=["test_compile_rocm"])
    def test_run_rocm_threshold(self):
        if self.compile_only:
            pytest.skip()
        return iree_run_module(
            VmfbManager.rocm_vmfb,
            device="hip",
            function="encode_prompts",
            args=[
                f"--parameters=model={self.real_weights.path}",
            ]
            + self.rocm_threshold_args
            + self.common_rule_flags,
        )