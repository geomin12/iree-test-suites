# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)

args = parser.parse_args()
model = args.model

for file in os.listdir(f"{Path.cwd()}/pkgci_test_suite/regression_tests/{model}"):
    # Ensure that only model layer json file is being read
    if ".json" in file:
        model_layer_name = file.split(".")[0]
        command = [
            "pytest pkgci_test_suite/regression_tests/test_model_threshold.py",
            "-rpfE",
            "--log-cli-level=info",
            "--capture=no",
            "--timeout=600",
            "--durations=0",
            f"--model-name {model}",
            f"--neural-net-name {model_layer_name}"
        ]
        subprocess.run(command)