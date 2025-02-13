import subprocess
import  os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str)
args = parser.parse_args()
model = args.model

for filename in os.listdir(f"./pkgci_test_suite/regression_tests/{model}"):
    if ".json" in filename:
        submodel_name = filename.split(".")[0]
        command = [
            "pytest",
            "./pkgci_test_suite/regression_tests/test_model_threshold.py",
            "-rpFe",
            "--log-cli-level=info",
            "--capture=no",
            "--timeout=600",
            "--durations=0",
            f"--model-name={model}",
            f"--submodel-name={submodel_name}"
        ]
        subprocess.run(command)