
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
config = AutoConfig.from_pretrained(model_id)

# For faster execution user can run with lesser layers, For Testing Purpose Only. Please ensure to use the configuration given below as random configurations may fail due to deepstack
config.vision_config.depth = 9
config.text_config.num_hidden_layers = 10
config.vision_config.deepstack_visual_indexes = [8]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    # For CCL activation
    # qaic_config={
    #     "ccl_enabled": True,
    # },
)


qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    continuous_batching=True,
    config=config,
    trust_remote_code=False,
)

passed_args = {
    "prefill_seq_len": 512,
    "ctx_len": 4096, #65536,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 4, # 32
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "prefill_only": True,
    "use_onnx_subfunctions": True,
    "retain_full_kv": True,
    "split_model_io": True,
    "enable_chunking": True,
    "skip_vision": True,
    "user_tiled": True,
    "node_precision_info": False,
    "qaic_config": {
        # "blocking_mode": "prefill_qkv",
        # "num_kv_blocks": 16,
        # "num_q_blocks": 8,
        # "skip_kv": True,
        # "n_rep_chunk": 1,
        "moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 256},
    },
    "mdp_strategy": "intersection",
    "connection_type": "mix",
    "allow_mxint8_mdp_io": True,
    "mdp_num_partitions": 2, # 4
}

qpc_path = qeff_model.compile(**passed_args)
print(f">>>>>>>>>>>>> qpc_path :{qpc_path}")
