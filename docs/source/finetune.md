# Finetune Infra

This repository provides the infrastructure for finetuning models using different hardware accelerators such as QAic.
Same CLI can be used to run finetuning on GPU by changing the value of device flag (for finetuning on GPU, install torch specific to CUDA).

## Installation

Same as QEfficient along with QAIC PyTorch Eager mode.

For QEfficient Library : https://github.com/quic/efficient-transformers

For torch_qaic, assuming QEfficient is already installed,
```bash
pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-0.1.0-cp312-cp312-linux_x86_64.whl
```
If qeff-env inside docker is used then torch_qaic and accelerate packages are already installed.

---

## Finetuning

Export the ENV variables to download and enable private datasets
```bash
export HF_DATASETS_TRUST_REMOTE_CODE=True
```

Export the ENV variables to get the device and HW traces and debugging logs
```bash
export QAIC_DEVICE_LOG_LEVEL=0 # For Device level logs
export QAIC_DEBUG=1 # To understand the CPU fallback ops
```
---

## Dataset Details

To download the Alpaca dataset, visit this [link](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json). Download the dataset and place it under the **dataset** directory. Make sure to update the training configuration accordingly.
```bash
wget -c https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json -P dataset/
```

To download the grammar dataset, visit this [link](https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/datasets/grammar_dataset/grammar_dataset_process.ipynb). Download the dataset and place it under the **datasets_grammar** directory. Make sure to update the training configuration accordingly.

---

## Usage

### Single SOC finetuning on QAIC

```python
python -m QEfficient.cloud.finetune --device qaic:0 --model_name "meta-llama/Llama-3.2-1B"
```
You can also configure various training parameters. Below is an example command line
```python
python -m QEfficient.cloud.finetune --device qaic:0 --use-peft --output_dir ./meta-sam --num_epochs 2 --context_length 256 
```

For more details on the usage of the training parameters, use the below command:
```python
python -m QEfficient.cloud.finetune -h
```
---

### Distributed training(DDP) on QAIC

```python
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune --device qaic --enable_ddp  --num_epochs 2  --model_name "meta-llama/Llama-3.2-1B"
```
**nproc-per-node is number of workers(QAIC devices) running locally.

---

### Multi Node(across multiple servers) finetuning on QAIC

This enables scaling training across multiple nodes.

Use servers with compatible/same network interface(eg:ethernet).

And supported only for linux servers now. Use servers connected to same switch for benefits in time while scaling.

```
PYTHONUNBUFFERED: make python prints unbuffered, especially useful to identify progress (or lack thereof) for distributed tasks.This is optional and not compulsory
```
```
GLOO_SOCKET_IFNAME: specify which network interface gloo (and indirectly qccl) uses for inter-host communication (eg: eno1, eth0 etc)
```
```
--nnodes: total number of hosts participating in the task
```
```
--nproc-per-node: number of processes launched on this host, usually coincides with number of accelerators on this host
```
```
--master_addr: ip of the host designated with node_rank=0 ($ ip addr)
```
```
--master_port: port on which host will be listening for other nodes to connect. (eg: 8888, 8000 etc)
```

Use --node-rank 0 on the host server and --node-rank 1 on client server(for dual server setup). When running distributed training across multiple servers, the --node-rank parameter must be assigned a unique value for each server, starting from 0 and incrementing by 1 for each additional server. For a setup with N servers it range from 0 to N-1.

Steps to run Multi Node Finetuning:

1. Launch Docker Containers on Each Node:

Run the following docker setup commands on both machines (server and client).

#### Expose QAIC accelerator devices

```
devices=(/dev/accel/*)
```

#### Start Docker container

```
sudo docker run -it \
    --name qaic_ddp1 \
    --net=host \
    --ipc=host \
    --add-host gb-292-blr-06:10.131.26.213 \
    --add-host gb-292-blr-30:10.131.30.207 \
    -v /home/ubuntu/:/home/ubuntu/ \
    "${devices[@]/#/--device=}" \
    docker-registry.qualcomm.com/qraniumtest/qranium:1.22.0.17-ubuntu22-x86_64 \
    /bin/bash
```
** Note :
In distributed ML setups, all nodes must resolve each other’s hostnames. If DNS in the environment does not resolve internal hostnames, we must manually force name resolution using --add-host.

2. Set QAIC Device Visibility

``` 
export QAIC_VISIBLE_DEVICES=$(seq -s, 0 63)

```

For example this sample command exposes devices 0–63 to the training process.

3. Activate the TORCH_QAIC Environment Inside the Container

```
source /opt/torch-qaic-env/bin/activate
```

4. Verify that the Qefficient Library is installed:

```
pip install -e .
```


5. Use below command on host server
```
QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune --device qaic --seed 0 --enable_ddp --num_epochs 2 --model_name "meta-llama/Llama-3.2-1B" --dataset gsm8k_dataset --output_dir training_results
```

6. Use below command on client server
```
QAIC_VISIBLE_DEVICES=0,1 GLOO_SOCKET_IFNAME=* torchrun --nnodes=2 --nproc-per-node=2 --node-rank=1 --master_addr=* --master_port=8888 -m QEfficient.cloud.finetune --device qaic --seed 0 --enable_ddp --num_epochs 2 --model_name "meta-llama/Llama-3.2-1B" --dataset gsm8k_dataset --output_dir training_results
```

---

## Visualization

Tensorboard logs are generated inside runs/ directory with date and time stamp.
to visualise the data,

```python
tensorboard --logdir runs/<file> --bind_all
```
---

## Some features/functionalities of fine-tuning stack:
    1) Gradient accumulation: By default, gradient accumulation happens for 4 steps. To update this value, command line argument gradient_accumulation_steps has to be passed. (Example: '--gradient_accumulation_steps 8')
    2) Gradient Checkpointing: By default, gradient checkpointing is disabled. To enable it, command line argument gradient_accumulation_steps has to be passed.


### 🔧 Steps to Fine-Tune with a Custom Dataset

1.  **Launching Fine-Tuning with a Custom Dataset**
    -   Use the following command-line arguments to begin fine-tuning using a custom dataset:
        ```bash
        --dataset custom_dataset --dataset_config data_config.json
        ```
    -   The `--dataset_config` argument is mandatory when `--dataset custom_dataset` is specified. The `data_config.json` file contains essential parameters used during dataset preprocessing.

        __Example `data_config.json` File__
        ```json
        {
        "train_split": "train",
        "test_split": "test",
        "test_split_ratio": 0.15,
        "preproc_file": "sample_dataset_preproc.py:preprocessing_fn",
        "collate_file": "sample_dataset_preproc.py:data_collate_fn",
        "disc_style": "sarcasm_more"
        }
        ```

2.  **Specifying the Preprocessing Function**
    -   In `data_config.json`, include a `"preproc_file"` mandatory key to define the path to your preprocessing Python file and the function within it.
    -   Use the format `"filename.py:function_name"`. The filename and function name both are required. 
        _Example:_
        ```json
        "preproc_file": "sample_dataset_preproc.py:preprocessing_fn"
        ```
    -   The preprocessing function must follow the structure below. The function parameters and the return type of the function should not be altered. The sample illustrates `apply_prompt_template` and `tokenize` as sub-functions, but we can define our own sub-functions as needed. For reference, check the example files in the [./QEfficient/finetune/dataset/](https://github.com/quic/efficient-transformers/tree/main/QEfficient/finetune/dataset) directory.
        ```python
        def preprocessing_fn(dataset_config, tokenizer, split, context_length=None):
            # Load the dataset or read from the disk
            # ...

            # Split the dataset into train and test splits if needed,
            # and use the appropriate split based on the 'split' argument.
            # ...

            def apply_prompt_template(example):
                # Apply prompt formatting to each datapoint (e.g., example)
                # ...
                return example # Return the processed example

            def tokenize(example):
                # Tokenize the formatted datapoint (e.g., example)
                # ...
                return tokenizer(example["text"], truncation=True, max_length=context_length) # Example tokenization

            # Apply prompt template to preprocess it in accordance to the dataset and task.
            dataset = dataset.map(apply_prompt_template, ...)

            # Finally, tokenize the dataset
            dataset = dataset.map(tokenize, batched=True, remove_columns=['text']) # Example batched tokenization

            # Each sample in the dataset should have keys acceptable by the HF
            # model and the loss function.
            # Typically, for CausalLM models used with 'generation' task_mode,
            # the keys should be 'input_ids', 'attention_mask', and 'labels'.
            return dataset
        ```
        -   In the sample preprocessing function above, the `split` variable takes its value from `data_config.json`. For the training dataset, the value will be taken from the `"train_split"` key, and for the evaluation/test dataset, it will be taken from the `"test_split"` key.
        -   Additional arguments needed for the preprocessing function can be passed in `data_config.json` and will be available via the `dataset_config` variable within the function. For instance, in the sample config above, `"test_split_ratio"` and `"disc_style"` keys can be used in the preprocessing function to define the test split ratio and style of the dataset. These values are accessed through the `dataset_config` variable. Check out the sample preprocessing file at [./QEfficient/finetune/dataset/custom_dataset/sample_dataset_preproc.py](https://github.com/quic/efficient-transformers/tree/main/QEfficient/finetune/dataset/custom_dataset/sample_dataset_preproc.py).

3.  **Custom Collate Function for Batching**
    -   When using a batch size greater than 1, we may need to override the default collate (batching different samples together in a batch) behavior by including a `"collate_file"` key in `data_config.json`.
    -   Use the same `"file.py:function"` format. If omitted, the default Hugging Face `DataCollatorForSeq2Seq` is typically used, which pads sequences to the longest length in the batch.
    -   A custom collate function must follow the structure below. The function parameters and the return type of the function should not be altered:
        ```python
        def get_data_collator(tokenizer):
            # Define and return a custom collate_fn here
            # ...
         
            # This function should take a list of samples and return a batch.
            # Example:
            # from transformers import DataCollatorForLanguageModeling
            # return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        ```
