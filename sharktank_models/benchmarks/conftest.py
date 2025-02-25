import pytest
from model_benchmark import ModelBenchmarkRunItem

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
        print(self.path)
