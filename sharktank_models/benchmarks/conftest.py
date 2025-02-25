import pytest
from model_benchmark import ModelBenchmarkRunItem
from pathlib import Path
from dataclasses import dataclass

THIS_DIR = Path(__file__).parent

def pytest_collect_file(parent, file_path):
    if file_path.suffix == ".json":
        return SharkTankModelBenchmarkTests.from_parent(parent, path=file_path)

@dataclass(frozen = True)
class BenchmarkTestSpec:
    model_name: str
    benchmark_file_name: str
    
class SharkTankModelBenchmarkTests(pytest.File):
    
    def collect(self):
        path = self.path.split("/")
        benchmark_file_name = path.split[-1].replace(".json", "")
        model_name = path.split[-2]
        
        item_name = f"{model_name} :: {benchmark_file_name}"
        
        spec = BenchmarkTestSpec(
            model_name = model_name,
            benchmark_file_name = benchmark_file_name
        )
        
        yield ModelBenchmarkRunItem.from_parent(self, name=item_name, spec=spec)
        