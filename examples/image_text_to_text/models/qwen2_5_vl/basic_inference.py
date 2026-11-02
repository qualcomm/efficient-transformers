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

## For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
config = AutoConfig.from_pretrained(model_id)
config.text_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

### use skip_vision=Ture, if want to run only text, ow false ###
skip_vision = True

if skip_vision:
    ## Only Text ##

    ## Set Batch_Size ##
    batch_size = 1
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=8,
        height=354,
        width=536,
        mxfp6_matmul=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
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
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)

else:
    batch_size = 1
    ## Vision + Text ##
    qeff_model.compile(
        batch_size=batch_size,
        prefill_seq_len=128,
        ctx_len=4096,
        num_cores=16,
        num_devices=8,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        mos=1,
    )

    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"

    image = Image.open(requests.get(image_url, stream=True).raw)

    messages_1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]

    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe about the color of the dog."},
            ],
        },
    ]

    messages = [messages_2] * batch_size

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
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)
