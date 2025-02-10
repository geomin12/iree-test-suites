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

class TestModelThreshold:

    @pytest.fixture(autouse = True, scope = "class")
    @classmethod
    def setup_class(self, pytestconfig):
        self.model_name = pytestconfig.getoption("model_name")
        self.neural_net_name = pytestconfig.getoption("neural_net_name")

        file_name = f"./{self.model_name}/{self.neural_net_name}.json"

        with open(file_name ,'r') as file:
            data = json.load(file)

            # retrieve source fixtures
            self.inputs = self.fetch_source_fixtures_for_run_flags(data.get("inputs"))
            self.outputs = self.fetch_source_fixtures_for_run_flags(data.get("outputs"))
            self.real_weights = self.fetch_source_fixture(data.get("real_weights"))
            self.mlir = self.fetch_source_fixture(data.get("mlir"))

            # setting custom compiler for cpu and rocm
            self.cpu_compiler_flags = data.get("cpu_compiler_flags")
            self.cpu_compiler_flags.append("--iree-hal-target-backends=llvm-cpu")
            self.rocm_compile_flags = data.get("rocm_compiler_flags")
            self.rocm_compile_flags.append("--iree-hal-target-backends=rocm")
            self.rocm_compile_flags.append(f"--iree-hip-target={rocm_chip}")

            self.common_rule_flags = self.common_run_flags_generation(self.inputs, self.outputs)
    
    def fetch_source_fixture(self, source_url):
        return fetch_source_fixture(source_url, group=f"{self.model_name}_{self.neural_net_name}") if source_url else None

    def fetch_source_fixtures_for_run_flags(self, inference_list):
        result = []
        for entry in inference_list:
            source = entry.get("source")
            value = entry.get("value")
            source_fixture = self.fetch_source_fixture(source)
            result.append([source_fixture.path, value])
        
        return result

    def common_run_flags_generation(self, input_list, output_list):
        flags_list = []

        for path, value in input_list:
            flags_list.append(f"--input={value}=@{path}")

        for path, value in output_list:
            flags_list.append(f"--expected_output={value}=@{path}")
        
        return flags_list


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
        iree_run_module(
            VmfbManager.cpu_vmfb,
            device="local-task",
            function="encode_prompts",
            args=[
                f"--parameters=model={self.real_weights.path}",
                "--expected_f16_threshold=1.0f",
            ]
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
        return iree_run_module(
            VmfbManager.rocm_vmfb,
            device="hip",
            function="encode_prompts",
            args=[
                f"--parameters=model={self.real_weights.path}",
                "--expected_f16_threshold=1.0f",
            ]
            + self.common_rule_flags,
        )