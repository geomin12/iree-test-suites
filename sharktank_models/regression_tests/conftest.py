# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from model_quality_run import ModelQualityRunItem
from pathlib import Path

THIS_DIR = Path(__file__).parent
backend = os.getenv("BACKEND", default="gfx942")
sku = os.getenv("SKU", default="mi300")

logger = logging.getLogger(__name__)

def pytest_configure():
    pytest.vmfb_manager = {}

def pytest_sessionstart(session):
    logger.info("Pytest quality test session is starting")

def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".json" and "regression_tests" in str(THIS_DIR):
        return SharkTankModelQualityTests.from_parent(parent, path=file_path)

@dataclass(frozen = True)
class QualityTestSpec:
    model_name: str
    submodel_name: str
    
class SharkTankModelQualityTests(pytest.File):
    
    def collect(self):
        path = str(self.path).split("/")
        submodel_name = path[-1].replace(".json", "")
        model_name = path[-2]
        
        item_name = f"{model_name} :: {submodel_name}"
        
        spec = QualityTestSpec(
            model_name = model_name,
            submodel_name = submodel_name
        )
        
        yield ModelQualityRunItem.from_parent(self, name=item_name, spec=spec)
    