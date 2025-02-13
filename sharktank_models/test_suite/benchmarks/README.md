## Regression tests

### Adding your own model

- To add your own model, create a directory under `benchmarks` and add JSON files that correspond to the submodels. Please follow the [JSON file schema in this README file](#required-and-optional-fields-for-the-json-model-file)

### How to run

- Example command to run all submodels for a specific model

```
tbd
```

- Example command to run a specific model and submodel

```
pytest sharktank_models/test_suite/benchmarks/test_model_benchmark.py \
    --log-cli-level=info \
    --timeout=600 \
    --retries 7 \
    --model-name sdxl \
    --submodel-name unet_fp16
```

### Required and optional fields for the JSON model file

| Field Name                       | Required | Type   | Description                                                                                                                  |
| -------------------------------- | -------- | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| inputs                           | required | array  | An array of input strings for the benchmark module (ex: `["1xi64, 1xf16]`)                                                   |
| function_run                     | required | string | The function that the `iree-benchmark-module` will run adnd benchmark                                                        |
| benchmark_repetitions            | required | float  | The number of times the benchmark tests will repeat                                                                          |
| benchmark_min_warmup_time        | required | float  | The minimum warm up time for the benchmark test                                                                              |
| golden_time_tolerance_multiplier | required | object | An object of tolerance multipliers, where the key is the sku and the value is the multiplier, (ex: `{"mi250": 1.3}`)         |
| golden_time                      | required | object | An object of golden times, where the key is the sku and the value is the golden time in ms, (ex: `{"mi250": 100}`)           |
| golden_dispatch                  | required | object | An object of golden dispatches, where the key is the sku and the value is the golden dispatch count, (ex: `{"mi250": 1602}`) |
| golden_size                      | required | object | An object of golden sizes, where the key is the sku and the value is the golden size in bytes, (ex: `{"mi250": 2000000}`)    |
| specific_rocm_chip_to_ignore     | optional | array  | An array of chip values, where the benchmark tests will ignore the chips specified                                           |

Please feel free to look at any JSON examples under a model directory (ex: sd3, sdxl)
