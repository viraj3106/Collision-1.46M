# COLLISION-10M Inference Benchmark Report

This report presents performance metrics of the COLLISION-10M model running sequentially on CPU.

## Model Setup
* **Model ID**: `collision-10m`
* **Checkpoint Hash**: `d256d46d962d6416fe22d2cfe80b13df0574279fb980d7d8576c2bdcf3775b97`
* **Model Loading & Warmup Time**: 0.534 seconds

## Performance Summary (10 Sequential Requests)
* **Prompt**: "What is machine learning?"
* **Settings**: max_tokens=100, temperature=0.7, top_k=50, top_p=0.9
* **Total Generated Tokens**: 975

| Metric | Value |
| :--- | :---: |
| **Average Latency** | 2317.6 ms |
| **Median Latency** | 2288.0 ms |
| **Minimum Latency** | 1952.8 ms |
| **Maximum Latency** | 2662.4 ms |
| **Average Throughput** | 42.38 tokens/sec |
| **Average RAM Usage** | 476.30 MB |
| **Peak RAM Usage** | 614.14 MB |

## Execution Logs

* **Request 1**: Latency: 2235.5 ms | Completion Tokens: 84 | Throughput: 37.99 tokens/sec | RAM: 337.5 MB
* **Request 2**: Latency: 2502.6 ms | Completion Tokens: 100 | Throughput: 40.09 tokens/sec | RAM: 369.1 MB
* **Request 3**: Latency: 2235.7 ms | Completion Tokens: 100 | Throughput: 44.84 tokens/sec | RAM: 400.1 MB
* **Request 4**: Latency: 2298.4 ms | Completion Tokens: 100 | Throughput: 43.62 tokens/sec | RAM: 430.6 MB
* **Request 5**: Latency: 1952.8 ms | Completion Tokens: 91 | Throughput: 46.75 tokens/sec | RAM: 461.2 MB
* **Request 6**: Latency: 2277.5 ms | Completion Tokens: 100 | Throughput: 44.03 tokens/sec | RAM: 491.9 MB
* **Request 7**: Latency: 2397.6 ms | Completion Tokens: 100 | Throughput: 41.81 tokens/sec | RAM: 521.8 MB
* **Request 8**: Latency: 2662.4 ms | Completion Tokens: 100 | Throughput: 37.66 tokens/sec | RAM: 553.1 MB
* **Request 9**: Latency: 2349.7 ms | Completion Tokens: 100 | Throughput: 42.68 tokens/sec | RAM: 583.7 MB
* **Request 10**: Latency: 2263.7 ms | Completion Tokens: 100 | Throughput: 44.32 tokens/sec | RAM: 614.1 MB
