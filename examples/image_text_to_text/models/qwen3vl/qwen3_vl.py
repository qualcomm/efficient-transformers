# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import requests
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "Qwen/Qwen3-VL-32B-Instruct"
config = AutoConfig.from_pretrained(model_id)

# config.vision_config.depth = 9
# config.text_config.num_hidden_layers = 1
# config.vision_config.deepstack_visual_indexes = [8]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    # # For CCL activation
    # qaic_config={
    #     "ccl_enabled": True,
    # },
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
### use skip_vision=Ture, if want to run only text, else false ###
skip_vision = True

# Compute-Context-Length (CCL) lists for prefill and decode. When both are None and
# ccl_enabled=True, they are auto-generated from ctx_len.
# comp_ctx_lengths_prefill = [2048]
# comp_ctx_lengths_decode = [65536]

if skip_vision:
    ## Only Text ##

    ## Set Batch_Size ##
    batch_size = 1
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        use_onnx_subfunctions=True,
        # comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        # comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]

    messages = [messages] * batch_size

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(processor.tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    batch_size = 1
    ## Vision + Text ##
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        split_model_io=True,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
        use_onnx_subfunctions=False,
        # comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        # comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"

    image = Image.open(requests.get(image_url, stream=True).raw)

    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Descibe the image in details."},
            ],
        },
    ]

    messages = [messages_1] * batch_size

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)
    streamer = TextStreamer(tokenizer)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(processor.tokenizer.batch_decode(output.generated_ids))
    print(output)
