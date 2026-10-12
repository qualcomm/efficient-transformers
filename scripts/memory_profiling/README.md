# QEfficient Memory Profiling

A memory profiling solution for QEfficient workflows with manual operation marking.



## Quick Start

```python
from profiler import QEffMemoryProfiler
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

# Initialize profiler with verbose output to see detailed memory tracking information
profiler = QEffMemoryProfiler(verbose=True)
# Start monitoring memory usage - this begins tracking memory consumption
profiler.start_monitoring()

# Mark the start of model loading operation for memory profiling, this will help to create stage wise partitioning the output graph
profiler.mark_operation("Loading model")

model = QEFFAutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Mark the export operation 
profiler.mark_operation("Export")
model.export()

# Mark the compilation operation 
profiler.mark_operation("Compile")
model.compile(prefill_seq_len=128, ctx_len=256, num_cores=16)

# Mark the text generation operation
profiler.mark_operation("Generation")
output = model.generate(prompts=["Hello world"], tokenizer=tokenizer, generation_len=100)

# Stop memory monitoring and generate reports
profiler.stop_monitoring()

# Print a detailed memory usage report to the console showing peak memory and operation-wise breakdown (optional)
print(profiler.get_memory_report())

# Generate a visual graph of memory usage over time and save it as an image file
profiler.generate_memory_graph("profile.png")
```

## Configuration

### Basic Configuration

```python
profiler = QEffMemoryProfiler(
    sampling_interval=0.1,           # Sample every 100ms
    output_file="my_profile.png",    # Custom output file
    verbose=True,                    # Enable detailed logging
    enable_cpu_monitoring=True,      # Monitor CPU usage
    enable_disk_monitoring=True,     # Monitor disk I/O
)
```

### Manual Operation Marking

```python
profiler = QEffMemoryProfiler()
profiler.start_monitoring()

# Manual operation marking
profiler.mark_operation("Custom Operation 1")
# ... your code ...

profiler.mark_operation("Custom Operation 2")
# ... more code ...

profiler.stop_monitoring()
```

## API Reference

### QEffMemoryProfiler

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_interval` | `float` | `0.05` | Time between samples (seconds) |
| `output_file` | `str` | `"qeff_memory_profile.png"` | Output file path |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `enable_cpu_monitoring` | `bool` | `True` | Monitor CPU usage |
| `enable_disk_monitoring` | `bool` | `True` | Monitor disk I/O |

#### Methods

- **`start_monitoring()`**: Start background monitoring
- **`stop_monitoring()`**: Stop monitoring and mark completion
- **`mark_operation(name: str)`**: Manually mark operation start
- **`get_memory_report() -> str`**: Generate comprehensive text report
- **`generate_memory_graph(filename: str)`**: Create visualization
- **`stop_and_save(filename: str) -> str`**: Convenience method to stop and save

#### Properties

- **`peak_rss`**: Peak RSS memory usage (MB)
- **`peak_operation`**: Operation during peak memory
- **`samples`**: List of collected profiling samples
- **`operations`**: List of marked operations with timestamps

## Operation Types

The profiler supports marking these common QEfficient operations:

- **Model Loading**: `from_pretrained`, `AutoModel`, `AutoTokenizer`
- **Export**: `model.export()`, ONNX transforms, PyTorch transforms
- **Compilation**: `model.compile()`, QNN compilation
- **Generation**: `model.generate()`, inference execution
- **Cleanup**: Memory cleanup, garbage collection

## Output

### Console Report
```
QEFFICIENT PERFORMANCE MONITORING REPORT
============================================================
Peak Memory Usage:
   • RSS (Physical): 18.7 GB at 14:23:45
   • Peak during:    Compilation

Memory Statistics:
   • Current RSS:    16.2 GB (Delta: +15.8 GB)
   • Duration:       185.3 seconds
   • Operations:     4

QEfficient Operations Timeline:
    1.    0.0s - Model Loading (25.2s) [+8.2 GB]
    2.   25.2s - Export (15.4s) [+2.1 GB]
    3.   40.6s - Compilation (120.8s) [+6.3 GB] <- Peak
    4.  161.4s - Generation (18.7s) [+1.2 GB]
```

### Visualization

The profiler generates a comprehensive 4-panel visualization:

1. **Memory Timeline**: RSS usage with colored operation phases
2. **CPU Usage**: CPU utilization with performance zones
3. **Disk I/O**: Read/write activity per operation phase
4. **Phase Duration**: Timing analysis with duration labels

#### Sample Output

![Sample Memory Profile](memory_profile_llama3.2.png)

*Example memory profiling output showing QEfficient workflow phases including model loading, ONNX transforms, compilation, and generation phases with detailed memory, CPU, and disk I/O metrics.*

## Advanced Usage


### Accessing Raw Data

```python
# Get synchronized data arrays
data = profiler.get_synchronized_data()
timestamps = data['timestamps']
memory_usage = data['rss_memory']
cpu_usage = data['cpu_usage']

# Access individual samples
for sample in profiler.samples:
    print(f"Time: {sample.timestamp}, RSS: {sample.rss_mb} MB")
```

## Integration Examples

### With Existing QEfficient Scripts

```python
# Add to existing QEfficient workflow
profiler = QEffMemoryProfiler(output_file="workflow_profile.png")
profiler.start_monitoring()

# Existing QEfficient code unchanged
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
# ... rest of workflow ...

# Add at end
report = profiler.stop_and_save()
print(report)
```


## Limitations

### Disk I/O Tracking

**Subprocess I/O Limitation**: Disk I/O tracking captures parent process I/O only. Subprocess I/O (e.g., compilation reading ONNX files via `subprocess.run()`) is not captured due to Linux I/O accounting limitations. During compilation phases, expect lower I/O readings than actual file operations performed by subprocesses.

## Compatibility

- **Python**: 3.7+
- **Dependencies**: `psutil`, `matplotlib`, `numpy`