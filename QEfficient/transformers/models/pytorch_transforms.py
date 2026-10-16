# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import gc
import warnings
from collections.abc import Callable
from types import MethodType

import torch
from torch import nn
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DisentangledSelfAttention,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconDecoderLayer,
    FalconForCausalLM,
    FalconModel,
)
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaModel,
    GemmaRMSNorm,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2Model,
    Gemma2RMSNorm,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3RMSNorm,
    Gemma3TextModel,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRouter,
    Gemma4VisionAttention,
)
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
    Glm4MoeTopkRouter,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeAttention,
    GPTBigCodeBlock,
    GPTBigCodeForCausalLM,
    GPTBigCodeModel,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssModel,
    GptOssRMSNorm,
)
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJForCausalLM, GPTJModel
from transformers.models.granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GraniteRMSNorm,
)
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeAttention,
    GraniteMoeDecoderLayer,
    GraniteMoeForCausalLM,
    GraniteMoeModel,
    GraniteMoeMoE,
    GraniteMoeRMSNorm,
    GraniteMoeRotaryEmbedding,
)

try:
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeParallelExperts,
        GraniteMoeTopKGating,
    )
except ImportError:
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeExperts as GraniteMoeParallelExperts,
    )
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeTopKRouter as GraniteMoeTopKGating,
    )
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4Router,
    Llama4TextAttention,
    Llama4TextDecoderLayer,
    Llama4TextExperts,
    Llama4TextModel,
    Llama4TextMoe,
    Llama4TextRMSNorm,
    Llama4VisionAttention,
    Llama4VisionModel,
)
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
)
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MistralRMSNorm,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
    Mistral3Model,
    Mistral3RMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaModel,
    MllamaRotaryEmbedding,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextModel,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
    MllamaVisionModel,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2RMSNorm,
)
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiForCausalLM, PhiModel
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3Model,
    Phi3RMSNorm,
)
from transformers.models.pixtral.modeling_pixtral import PixtralRMSNorm, PixtralVisionModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2RMSNorm,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm as Qwen2_5RMSNorm
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm as Qwen2_5RMSNorm
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.nomic_bert.modeling_nomic_bert import NomicBertEmbeddings
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextModel,
    Qwen3_5VisionAttention,
    Qwen3_5VisionModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTopKRouter,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeTopKRouter,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeTextDecoderLayer,
    Qwen3VLMoeTextExperts,
    Qwen3VLMoeTextModel,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeTextTopKRouter,
    Qwen3VLMoeVisionAttention,
    Qwen3VLMoeVisionModel,
)
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5LayerNorm,
    T5Stack,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPositionalEmbedding,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings, XLMRobertaModel

from QEfficient.base.pytorch_transforms import (
    ExternalModuleMapperTransform,
    ModuleMappingTransform,
    ModuleMutatorTransform,
    PytorchTransform,
)
from QEfficient.customop import CustomRMSNormAIC, GemmaCustomRMSNormAIC
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP, PooledModel, validate_user_pooling_function
from QEfficient.transformers.models.bert.modeling_bert import (
    QEffBertModel,
    QEffNomicBertEmbeddings,
    QEffRobertaEmbeddings,
    QEffRobertaModel,
    QEffXLMRobertaEmbeddings,
    QEffXLMRobertaModel,
)
from QEfficient.transformers.models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenBlock,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from QEfficient.transformers.models.deberta_v2.modeling_deberta_v2 import (
    QEffDisentangledSelfAttention,
)
from QEfficient.transformers.models.deepseek_v3.modeling_deepseek import (
    QEffDeepseekV3Attention,
    QEffDeepseekV3CustomRMSNormAIC,
    QEffDeepseekV3DecoderLayer,
    QEffDeepseekV3ForCausalLM,
    QEffDeepseekV3Model,
    QEffDeepseekV3MoE,
)
from QEfficient.transformers.models.dflash_draft.modeling_dflash_draft import (
    QEffDFlashAttention,
    QEffDFlashDecoderLayer,
    QEffDFlashForCausalLM,
    QEffDFlashModel,
)
from QEfficient.transformers.models.falcon.modeling_falcon import (
    QEffFalconAttention,
    QEffFalconDecoderLayer,
    QEffFalconForCausalLM,
    QEffFalconModel,
)
from QEfficient.transformers.models.gemma.modeling_gemma import (
    QEffGemmaAttention,
    QEffGemmaDecoderLayer,
    QEffGemmaForCausalLM,
    QEffGemmaModel,
)
from QEfficient.transformers.models.gemma2.modeling_gemma2 import (
    QEffGemma2Attention,
    QEffGemma2DecoderLayer,
    QEffGemma2ForCausalLM,
    QEffGemma2Model,
)
from QEfficient.transformers.models.gemma3.modeling_gemma3 import (
    QEffGemma3Attention,
    QEffGemma3CustomRMSNormAIC,
    QEffGemma3DecoderLayer,
    QEffGemma3DecoderWrapper,
    QEffGemma3ForCausalLMModel,
    QEffGemma3ForConditionalGeneration,
    QEffGemma3TextModel,
)
from QEfficient.transformers.models.gemma4.modeling_gemma4 import (
    QEffGemma4ClippableLinear,
    QEffGemma4CustomRMSNormAIC,
    QEffGemma4ForCausalLM,
    QEffGemma4ForConditionalGeneration,
    QEffGemma4TextAttention,
    QEffGemma4TextDecoderLayer,
    QEffGemma4TextModel,
    QEffGemma4TextRouter,
    QEffGemma4VisionAttention,
)
from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import (
    QEffGlm4MoeAttention,
    QEffGlm4MoeDecoderLayer,
    QEffGlm4MoeForCausalLM,
    QEffGlm4MoeModel,
    QEffGlm4MoeMoE,
    QEffGlm4MoeRotaryEmbedding,
    QEffGlm4MoeTopkRouter,
)
from QEfficient.transformers.models.gpt2.modeling_gpt2 import (
    QEffGPT2Attention,
    QEffGPT2Block,
    QEffGPT2LMHeadModel,
    QEffGPT2Model,
)
from QEfficient.transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    QEffGPTBigCodeAttention,
    QEffGPTBigCodeBlock,
    QEffGPTBigCodeForCausalLM,
    QEffGPTBigCodeModel,
)
from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
    QEffGptOssAttention,
    QEffGptOssDecoderLayer,
    QEffGptOssExperts,
    QEffGptOssForCausalLM,
    QEffGptOssMLP,
    QEffGptOssModel,
    QEffPrefillOnlyChunkedGptOssAttention,
    QEffPrefillOnlyGptOssAttention,
    QEffPrefillOnlyGptOssModel,
)
from QEfficient.transformers.models.gptj.modeling_gptj import (
    QEffGPTJAttention,
    QEffGPTJBlock,
    QEffGPTJForCausalLM,
    QEffGPTJModel,
)
from QEfficient.transformers.models.granite.modeling_granite import (
    QEffGraniteAttention,
    QEffGraniteDecoderLayer,
    QEffGraniteForCausalLM,
    QEffGraniteModel,
)
from QEfficient.transformers.models.granitemoe.modeling_granitemoe import (
    QEffGraniteMoeAttention,
    QEffGraniteMoeDecoderLayer,
    QEffGraniteMoeForCausalLM,
    QEffGraniteMoeModel,
    QEffGraniteMoeMoE,
    QEffGraniteMoeParallelExperts,
    QEffGraniteMoeRotaryEmbedding,
    QEffGraniteMoeTopKGating,
)
from QEfficient.transformers.models.grok_1.modeling_grok1 import (
    QEFFGrok1CustomRMSNormAIC,
    QEffGrok1DecoderLayer,
    QEffGrok1Model,
    QEffGrok1ModelForCausalLM,
    QEffGrok1MoeBlock,
    QEffGrok1MultiHeadAttention,
)
from QEfficient.transformers.models.internvl.modeling_internvl import (
    QEffInternDecoderWrapper,
    QEffInternVisionEmbeddings,
    QEffInternVLModel,
)
from QEfficient.transformers.models.kimi_k25.modeling_kimi_k25 import (
    QEffKimiK25ForConditionalGeneration,
    QEffLearnable2DInterpPosEmbDivided_fixed,
    QEffMoonViT3dEncoder,
)
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
    QEffLlamaRotaryEmbedding,
)
from QEfficient.transformers.models.llama4.modeling_llama4 import (
    QEffLlama4DecoderWrapper,
    QEffLlama4ForCausalLM,
    QEffLlama4ForConditionalGeneration,
    QEffLlama4Router,
    QEffLlama4TextAttention,
    QEffLlama4TextDecoderLayer,
    QEffLlama4TextExperts,
    QEffLlama4TextModel,
    QEffLlama4TextMoe,
    QEffLlama4VisionAttention,
    QEffLlama4VisionModel,
)
from QEfficient.transformers.models.llava.modeling_llava import (
    QEFFLlavaDecoderWrapper,
    QEffLlavaForConditionalGeneration,
)
from QEfficient.transformers.models.llava_next.modeling_llava_next import (
    QEffLlavaNextDecoderWrapper,
    QEffLlavaNextForConditionalGeneration,
)
from QEfficient.transformers.models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from QEfficient.transformers.models.mistral3.modeling_mistral3 import (
    QEffMistral3ForConditionalGeneration,
    QEffMistral3Model,
    QEffPixtralVisionModel,
)
from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QeffMixtralDecoderLayer,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from QEfficient.transformers.models.mllama.modeling_mllama import (
    QEffMllamaCrossAttentionDecoderLayer,
    QEffMllamaForCausalLM,
    QEffMllamaForConditionalGeneration,
    QEffMllamaModel,
    QEffMllamaRotaryEmbedding,
    QEffMllamaSelfAttentionDecoderLayer,
    QEffMllamaTextCrossAttentionSingleQPC,
    QEffMllamaTextCrossAttentionTwoQPC,
    QEffMllamaTextModel,
    QEffMllamaTextSelfAttention,
    QEffMllamaVisionModel,
)
from QEfficient.transformers.models.molmo.modeling_molmo import (
    QEffMolmo,
    QEffMolmoBlock,
    QEffMolmoModel,
    QEffMolmoSequentialBlock,
    QEffMultiHeadDotProductAttention,
)
from QEfficient.transformers.models.mpt.modeling_mpt import (
    QEffMptAttention,
    QEffMptBlock,
    QEffMptForCausalLM,
    QEFfMptModel,
)
from QEfficient.transformers.models.olmo2.modeling_olmo2 import (
    QEffOlmo2Attention,
    QEffOlmo2DecoderLayer,
    QEffOlmo2ForCausalLM,
    QEffOlmo2Model,
)
from QEfficient.transformers.models.phi.modeling_phi import (
    QEffPhiAttention,
    QEffPhiDecoderLayer,
    QEffPhiForCausalLM,
    QEffPhiModel,
)
from QEfficient.transformers.models.phi3.modeling_phi3 import (
    QEffPhi3Attention,
    QEffPhi3DecoderLayer,
    QEffPhi3ForCausalLM,
    QEffPhi3Model,
)
from QEfficient.transformers.models.qwen2.modeling_qwen2 import (
    QEffQwen2Attention,
    QEffQwen2DecoderLayer,
    QEffQwen2ForCausalLM,
    QEffQwen2Model,
)
from QEfficient.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    QEffQwen2_5_VisionTransformerPretrainedModel,
    QEffQwen2_5_VLAttention,
    QEffQwen2_5_VLDecoderLayer,
    QEffQwen2_5_VLModel,
    QEffQwen2_5_VLTextModel,
    QEffQwen2_5_VLVisionAttention,
    QEffQwen2_5_VLVisionBlock,
    QEffQwen_2_5_vl_DecoderWrapper,
    QEffQwen_2_5_vl_ForConditionalGeneration,
)
from QEfficient.transformers.models.qwen3.modeling_qwen3 import (
    QEffQwen3Attention,
    QEffQwen3DecoderLayer,
    QEffQwen3ForCausalLM,
    QEffQwen3Model,
)
from QEfficient.transformers.models.qwen3_5.modeling_qwen3_5 import (
    QEffQwen3_5Attention,
    QEffQwen3_5DecoderLayer,
    QEffQwen3_5ForCausalLM,
    QEffQwen3_5ForConditionalGeneration,
    QEffQwen3_5GatedDeltaNet,
    QEffQwen3_5GatedDeltaNetCustomRMSNormAIC,
    QEffQwen3_5Model,
    QEffQwen3_5TextModel,
    QEffQwen3_5VisionAttention,
    QEffQwen3_5VisionModel,
)
from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    QEffQwen3_5MoeAttention,
    QEffQwen3_5MoeDecoderLayer,
    QEffQwen3_5MoeExperts,
    QEffQwen3_5MoeForCausalLM,
    QEffQwen3_5MoeForConditionalGeneration,
    QEffQwen3_5MoeGatedDeltaNet,
    QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC,
    QEffQwen3_5MoeModel,
    QEffQwen3_5MoeSparseMoeBlock,
    QEffQwen3_5MoeTextModel,
    QEffQwen3_5MoeTopKRouter,
    QEffQwen3_5MoeVisionAttention,
    QEffQwen3_5MoeVisionModel,
)
from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import (
    QEffQwen3MoeAttention,
    QEffQwen3MoeDecoderLayer,
    QEffQwen3MoeExperts,
    QEffQwen3MoeForCausalLM,
    QEffQwen3MoeModel,
    QEffQwen3MoeRotaryEmbedding,
    QEffQwen3MoeSparseMoeBlock,
    QEffQwen3MoeTopKRouter,
)
from QEfficient.transformers.models.qwen3_vl.modeling_qwen3_vl import (
    QEffQwen3VLForConditionalGeneration,
    QEffQwen3VLModel,
    QEffQwen3VLTextAttention,
    QEffQwen3VLTextDecoderLayer,
    QEffQwen3VLTextModel,
    QEffQwen3VLTextRotaryEmbedding,
    QEffQwen3VLVisionAttention,
    QEffQwen3VLVisionModel,
)
from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    QEffQwen3VLMoeForConditionalGeneration,
    QEffQwen3VLMoeModel,
    QEffQwen3VLMoeTextAttention,
    QEffQwen3VLMoeTextDecoderLayer,
    QEffQwen3VLMoeTextExperts,
    QEffQwen3VLMoeTextModel,
    QEffQwen3VLMoeTextRotaryEmbedding,
    QEffQwen3VLMoeTextSparseMoeBlock,
    QEffQwen3VLMoeTextTopKRouter,
    QEffQwen3VLMoeVisionAttention,
    QEffQwen3VLMoeVisionModel,
)
from QEfficient.transformers.models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEFFStarcoder2DecoderLayer,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)
from QEfficient.transformers.models.t5.modeling_t5 import QEffT5Attention, QEffT5LayerNorm, QEffT5Stack
from QEfficient.transformers.models.wav2vec2.modeling_wav2vec2 import (
    QEffWav2Vec2Encoder,
    QEffWav2Vec2EncoderStableLayerNorm,
)
from QEfficient.transformers.models.whisper.modeling_whisper import (
    QEffWhisperAttention,
    QEffWhisperDecoder,
    QEffWhisperDecoderLayer,
    QEffWhisperEncoder,
    QEffWhisperForConditionalGeneration,
    QEffWhisperModel,
    QEffWhisperPositionalEmbedding,
)
from QEfficient.transformers.moe import (
    MoEFlavour,
    QEffMoEBlockMixin,
    pack_moe_weights_for_expert_parallel,
    unpack_moe_weights_from_expert_parallel,
)
from QEfficient.transformers.post_processing import build_and_attach_mlp, model_type_registry
from QEfficient.transformers.sampler.sampler import sampler_forward
from QEfficient.transformers.spd.spd_transform_forward import tlm_forward
from QEfficient.utils.checkpoint_utils import load_checkpoint_weights
from QEfficient.utils.config_utils import (
    resolve_attention_heads,
    resolve_hidden_size,
    resolve_kv_heads,
    set_kv_head_aliases,
)
from QEfficient.utils.constants import (
    _DFLASH_TARGET_ABSMAX,
    ATTENTION_HEAD_CONFIG_KEYS,
    DEFAULT_AIC_NUM_CORES,
    HIDDEN_SIZE_CONFIG_KEYS,
    KV_HEAD_CONFIG_KEYS,
    MOE_PREFILL_PACKED_CHUNK_SIZE,
    ONNX_EXPORT_EXAMPLE_SEQ_LEN,
)
from QEfficient.utils.logging_utils import QEFFLogger
from QEfficient.utils.repeat_kv_utils import (
    duplicate_kv_projection_weights,
    get_attention_module,
    get_projection_layer,
    get_text_model,
    is_mla_model,
)

logger = QEFFLogger.get_logger("MODEL")
SPD_TARGET = "target"


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        GemmaRMSNorm: GemmaCustomRMSNormAIC,
        Gemma2RMSNorm: GemmaCustomRMSNormAIC,
        GptOssRMSNorm: CustomRMSNormAIC,
        LlamaRMSNorm: CustomRMSNormAIC,
        Llama4TextRMSNorm: CustomRMSNormAIC,
        MistralRMSNorm: CustomRMSNormAIC,
        Mistral3RMSNorm: CustomRMSNormAIC,
        MixtralRMSNorm: CustomRMSNormAIC,
        Phi3RMSNorm: CustomRMSNormAIC,
        Qwen2RMSNorm: CustomRMSNormAIC,
        Qwen3RMSNorm: CustomRMSNormAIC,
        Qwen2_5RMSNorm: CustomRMSNormAIC,
        MllamaTextRMSNorm: CustomRMSNormAIC,
        GraniteRMSNorm: CustomRMSNormAIC,
        PixtralRMSNorm: CustomRMSNormAIC,
        GraniteMoeRMSNorm: CustomRMSNormAIC,
        Qwen3MoeRMSNorm: CustomRMSNormAIC,
        Gemma3RMSNorm: QEffGemma3CustomRMSNormAIC,
        Gemma4RMSNorm: QEffGemma4CustomRMSNormAIC,
        Olmo2RMSNorm: CustomRMSNormAIC,
        Qwen3VLMoeTextRMSNorm: CustomRMSNormAIC,
        Qwen3VLTextRMSNorm: CustomRMSNormAIC,
        Glm4MoeRMSNorm: CustomRMSNormAIC,
        Wav2Vec2Encoder: QEffWav2Vec2Encoder,
        Wav2Vec2EncoderStableLayerNorm: QEffWav2Vec2EncoderStableLayerNorm,
        # BERT-family: replace _create_attention_masks (uses create_bidirectional_mask,
        # which breaks ONNX tracing) with an ONNX-safe _prepare_4d_attention_mask version.
        BertModel: QEffBertModel,
        RobertaModel: QEffRobertaModel,
        XLMRobertaModel: QEffXLMRobertaModel,
        # *Embeddings: fix a FakeTensor/meta-device mismatch in the buffered
        # token_type_ids gather (see QEff*Embeddings docstrings in modeling_bert.py).
        RobertaEmbeddings: QEffRobertaEmbeddings,
        XLMRobertaEmbeddings: QEffXLMRobertaEmbeddings,
        NomicBertEmbeddings: QEffNomicBertEmbeddings,
        Qwen3_5RMSNorm: GemmaCustomRMSNormAIC,
        Qwen3_5MoeRMSNorm: GemmaCustomRMSNormAIC,
        Qwen3_5RMSNormGated: QEffQwen3_5GatedDeltaNetCustomRMSNormAIC,
        Qwen3_5MoeRMSNormGated: QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC,
    }


class KVCacheTransform(ModuleMappingTransform):
    _module_mapping = {
        # GLMMoe
        Glm4MoeModel: QEffGlm4MoeModel,
        Glm4MoeForCausalLM: QEffGlm4MoeForCausalLM,
        Glm4MoeAttention: QEffGlm4MoeAttention,
        Glm4MoeDecoderLayer: QEffGlm4MoeDecoderLayer,
        Glm4MoeRotaryEmbedding: QEffGlm4MoeRotaryEmbedding,
        # CodeGen
        CodeGenAttention: QEffCodeGenAttention,
        CodeGenBlock: QEffCodeGenBlock,
        CodeGenModel: QEffCodeGenModel,
        CodeGenForCausalLM: QEffCodeGenForCausalLM,
        # Falcon
        FalconAttention: QEffFalconAttention,
        FalconDecoderLayer: QEffFalconDecoderLayer,
        FalconModel: QEffFalconModel,
        FalconForCausalLM: QEffFalconForCausalLM,
        # GPT2
        GPT2Attention: QEffGPT2Attention,
        GPT2Block: QEffGPT2Block,
        GPT2Model: QEffGPT2Model,
        GPT2LMHeadModel: QEffGPT2LMHeadModel,
        # GPTJ
        GPTJAttention: QEffGPTJAttention,
        GPTJBlock: QEffGPTJBlock,
        GPTJModel: QEffGPTJModel,
        GPTJForCausalLM: QEffGPTJForCausalLM,
        # Llama
        LlamaAttention: QEffLlamaAttention,
        LlamaDecoderLayer: QEffLlamaDecoderLayer,
        LlamaModel: QEffLlamaModel,
        LlamaForCausalLM: QEffLlamaForCausalLM,
        LlamaRotaryEmbedding: QEffLlamaRotaryEmbedding,
        # Llama4
        Llama4TextAttention: QEffLlama4TextAttention,
        Llama4ForCausalLM: QEffLlama4ForCausalLM,
        Llama4TextDecoderLayer: QEffLlama4TextDecoderLayer,
        Llama4TextModel: QEffLlama4TextModel,
        Llama4ForConditionalGeneration: QEffLlama4ForConditionalGeneration,
        Llama4VisionAttention: QEffLlama4VisionAttention,
        Llama4VisionModel: QEffLlama4VisionModel,
        # Llava
        LlavaForConditionalGeneration: QEffLlavaForConditionalGeneration,
        # Llava Next
        LlavaNextForConditionalGeneration: QEffLlavaNextForConditionalGeneration,
        # Gemma
        GemmaAttention: QEffGemmaAttention,
        GemmaDecoderLayer: QEffGemmaDecoderLayer,
        GemmaModel: QEffGemmaModel,
        GemmaForCausalLM: QEffGemmaForCausalLM,
        # Qwen3Moe
        Qwen3MoeForCausalLM: QEffQwen3MoeForCausalLM,
        Qwen3MoeModel: QEffQwen3MoeModel,
        Qwen3MoeDecoderLayer: QEffQwen3MoeDecoderLayer,
        Qwen3MoeAttention: QEffQwen3MoeAttention,
        Qwen3MoeRotaryEmbedding: QEffQwen3MoeRotaryEmbedding,
        # Qwen3VLMoe
        Qwen3VLMoeForConditionalGeneration: QEffQwen3VLMoeForConditionalGeneration,
        Qwen3VLMoeModel: QEffQwen3VLMoeModel,
        Qwen3VLMoeTextAttention: QEffQwen3VLMoeTextAttention,
        Qwen3VLMoeTextDecoderLayer: QEffQwen3VLMoeTextDecoderLayer,
        Qwen3VLMoeVisionAttention: QEffQwen3VLMoeVisionAttention,
        Qwen3VLMoeVisionModel: QEffQwen3VLMoeVisionModel,
        Qwen3VLMoeTextModel: QEffQwen3VLMoeTextModel,
        Qwen3VLMoeTextRotaryEmbedding: QEffQwen3VLMoeTextRotaryEmbedding,
        # Qwen3vl
        Qwen3VLForConditionalGeneration: QEffQwen3VLForConditionalGeneration,
        Qwen3VLModel: QEffQwen3VLModel,
        Qwen3VLTextAttention: QEffQwen3VLTextAttention,
        Qwen3VLTextDecoderLayer: QEffQwen3VLTextDecoderLayer,
        Qwen3VLVisionAttention: QEffQwen3VLVisionAttention,
        Qwen3VLVisionModel: QEffQwen3VLVisionModel,
        Qwen3VLTextModel: QEffQwen3VLTextModel,
        Qwen3VLTextRotaryEmbedding: QEffQwen3VLTextRotaryEmbedding,
        # Gemma2
        Gemma2Attention: QEffGemma2Attention,
        Gemma2DecoderLayer: QEffGemma2DecoderLayer,
        Gemma2Model: QEffGemma2Model,
        Gemma2ForCausalLM: QEffGemma2ForCausalLM,
        Gemma3Attention: QEffGemma3Attention,
        Gemma3DecoderLayer: QEffGemma3DecoderLayer,
        Gemma3TextModel: QEffGemma3TextModel,
        Gemma3ForCausalLM: QEffGemma3ForCausalLMModel,
        Gemma3ForConditionalGeneration: QEffGemma3ForConditionalGeneration,
        # Gemma4
        Gemma4TextAttention: QEffGemma4TextAttention,
        Gemma4TextDecoderLayer: QEffGemma4TextDecoderLayer,
        Gemma4TextModel: QEffGemma4TextModel,
        Gemma4ForCausalLM: QEffGemma4ForCausalLM,
        Gemma4ForConditionalGeneration: QEffGemma4ForConditionalGeneration,
        Gemma4VisionAttention: QEffGemma4VisionAttention,
        Gemma4ClippableLinear: QEffGemma4ClippableLinear,
        # GPT_OSS
        GptOssAttention: QEffGptOssAttention,
        GptOssDecoderLayer: QEffGptOssDecoderLayer,
        GptOssModel: QEffGptOssModel,
        GptOssForCausalLM: QEffGptOssForCausalLM,
        # Granite
        GraniteModel: QEffGraniteModel,
        GraniteForCausalLM: QEffGraniteForCausalLM,
        GraniteAttention: QEffGraniteAttention,
        GraniteDecoderLayer: QEffGraniteDecoderLayer,
        # GraniteMoe
        GraniteMoeModel: QEffGraniteMoeModel,
        GraniteMoeForCausalLM: QEffGraniteMoeForCausalLM,
        GraniteMoeAttention: QEffGraniteMoeAttention,
        GraniteMoeRotaryEmbedding: QEffGraniteMoeRotaryEmbedding,
        GraniteMoeDecoderLayer: QEffGraniteMoeDecoderLayer,
        # mllama
        MllamaTextRMSNorm: CustomRMSNormAIC,
        MllamaTextSelfAttention: QEffMllamaTextSelfAttention,
        MllamaSelfAttentionDecoderLayer: QEffMllamaSelfAttentionDecoderLayer,
        MllamaModel: QEffMllamaModel,
        MllamaCrossAttentionDecoderLayer: QEffMllamaCrossAttentionDecoderLayer,
        MllamaRotaryEmbedding: QEffMllamaRotaryEmbedding,
        MllamaVisionModel: QEffMllamaVisionModel,
        MllamaTextModel: QEffMllamaTextModel,
        MllamaForCausalLM: QEffMllamaForCausalLM,
        MllamaForConditionalGeneration: QEffMllamaForConditionalGeneration,
        # Mistral
        MistralAttention: QEffMistralAttention,
        MistralDecoderLayer: QEffMistralDecoderLayer,
        MistralModel: QEffMistralModel,
        MistralForCausalLM: QEffMistralForCausalLM,
        # Mistral3
        Mistral3ForConditionalGeneration: QEffMistral3ForConditionalGeneration,
        Mistral3Model: QEffMistral3Model,
        # Mixtral
        MixtralAttention: QEffMixtralAttention,
        MixtralDecoderLayer: QeffMixtralDecoderLayer,
        MixtralModel: QEffMixtralModel,
        MixtralForCausalLM: QEffMixtralForCausalLM,
        # Mpt
        MptAttention: QEffMptAttention,
        MptBlock: QEffMptBlock,
        MptModel: QEFfMptModel,
        MptForCausalLM: QEffMptForCausalLM,
        # Phi3
        Phi3Attention: QEffPhi3Attention,
        Phi3DecoderLayer: QEffPhi3DecoderLayer,
        Phi3Model: QEffPhi3Model,
        Phi3ForCausalLM: QEffPhi3ForCausalLM,
        # Phi
        PhiAttention: QEffPhiAttention,
        PhiDecoderLayer: QEffPhiDecoderLayer,
        PhiModel: QEffPhiModel,
        PhiForCausalLM: QEffPhiForCausalLM,
        # Pixtral
        PixtralVisionModel: QEffPixtralVisionModel,
        # Qwen2
        Qwen2Attention: QEffQwen2Attention,
        Qwen2DecoderLayer: QEffQwen2DecoderLayer,
        Qwen2Model: QEffQwen2Model,
        Qwen2ForCausalLM: QEffQwen2ForCausalLM,
        # Qwen3
        Qwen3Attention: QEffQwen3Attention,
        Qwen3DecoderLayer: QEffQwen3DecoderLayer,
        Qwen3Model: QEffQwen3Model,
        Qwen3ForCausalLM: QEffQwen3ForCausalLM,
        # Qwen3_5
        Qwen3_5GatedDeltaNet: QEffQwen3_5GatedDeltaNet,
        Qwen3_5DecoderLayer: QEffQwen3_5DecoderLayer,
        Qwen3_5TextModel: QEffQwen3_5TextModel,
        Qwen3_5Model: QEffQwen3_5Model,
        Qwen3_5ForCausalLM: QEffQwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration: QEffQwen3_5ForConditionalGeneration,
        Qwen3_5Attention: QEffQwen3_5Attention,
        Qwen3_5VisionAttention: QEffQwen3_5VisionAttention,
        Qwen3_5VisionModel: QEffQwen3_5VisionModel,
        # Qwen3_5_Moe
        Qwen3_5MoeGatedDeltaNet: QEffQwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeDecoderLayer: QEffQwen3_5MoeDecoderLayer,
        Qwen3_5MoeTextModel: QEffQwen3_5MoeTextModel,
        Qwen3_5MoeModel: QEffQwen3_5MoeModel,
        Qwen3_5MoeForConditionalGeneration: QEffQwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeForCausalLM: QEffQwen3_5MoeForCausalLM,
        Qwen3_5MoeAttention: QEffQwen3_5MoeAttention,
        Qwen3_5MoeVisionAttention: QEffQwen3_5MoeVisionAttention,
        Qwen3_5MoeVisionModel: QEffQwen3_5MoeVisionModel,
        # Qwen2.5 VL
        Qwen2_5_VLForConditionalGeneration: QEffQwen_2_5_vl_ForConditionalGeneration,
        Qwen2_5_VLModel: QEffQwen2_5_VLModel,
        Qwen2_5_VLAttention: QEffQwen2_5_VLAttention,
        Qwen2_5_VLDecoderLayer: QEffQwen2_5_VLDecoderLayer,
        Qwen2_5_VisionTransformerPretrainedModel: QEffQwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLVisionAttention: QEffQwen2_5_VLVisionAttention,
        Qwen2_5_VLVisionBlock: QEffQwen2_5_VLVisionBlock,
        Qwen2_5_VLTextModel: QEffQwen2_5_VLTextModel,
        # Starcoder2
        Starcoder2Attention: QEffStarcoder2Attention,
        Starcoder2DecoderLayer: QEFFStarcoder2DecoderLayer,
        Starcoder2Model: QEffStarcoder2Model,
        Starcoder2ForCausalLM: QEffStarcoder2ForCausalLM,
        # GptBigcode
        GPTBigCodeAttention: QEffGPTBigCodeAttention,
        GPTBigCodeBlock: QEffGPTBigCodeBlock,
        GPTBigCodeModel: QEffGPTBigCodeModel,
        GPTBigCodeForCausalLM: QEffGPTBigCodeForCausalLM,
        # Olmo2
        Olmo2Attention: QEffOlmo2Attention,
        Olmo2DecoderLayer: QEffOlmo2DecoderLayer,
        Olmo2Model: QEffOlmo2Model,
        Olmo2ForCausalLM: QEffOlmo2ForCausalLM,
        # Whisper encoder and decoder layers
        WhisperPositionalEmbedding: QEffWhisperPositionalEmbedding,
        WhisperAttention: QEffWhisperAttention,
        WhisperDecoderLayer: QEffWhisperDecoderLayer,
        WhisperEncoder: QEffWhisperEncoder,
        WhisperDecoder: QEffWhisperDecoder,
        WhisperModel: QEffWhisperModel,
        WhisperForConditionalGeneration: QEffWhisperForConditionalGeneration,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed


class PrefillOnlyTransform(ModuleMappingTransform):
    _module_mapping = {
        QEffGptOssModel: QEffPrefillOnlyGptOssModel,
        QEffGptOssAttention: QEffPrefillOnlyGptOssAttention,
    }


class PrefillOnlyChunkedTransform(ModuleMappingTransform):
    _module_mapping = {
        # GPT_OSS
        QEffGptOssModel: QEffPrefillOnlyGptOssModel,
        QEffGptOssAttention: QEffPrefillOnlyChunkedGptOssAttention,
    }


class RevertPrefillKeepAttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        # GPT_OSS
        QEffGptOssModel: QEffPrefillOnlyGptOssModel,
        QEffPrefillOnlyGptOssAttention: QEffPrefillOnlyChunkedGptOssAttention,
        QEffGptOssAttention: QEffPrefillOnlyChunkedGptOssAttention,
    }


class RevertPrefillOnlyTransform(ModuleMappingTransform):
    _module_mapping = {
        **{v: k for k, v in PrefillOnlyTransform._module_mapping.items()},
        **{v: k for k, v in PrefillOnlyChunkedTransform._module_mapping.items()},
    }


class ReplicateKVHeadTransform(ModuleMutatorTransform):
    """
    Replicates KV heads in attention modules to match the number of KV heads in the target model.
    This transform is used when the source model has fewer KV heads than required in target model.
    """

    @classmethod
    def mutate(
        cls,
        original_module: nn.Module,
        parent_module: nn.Module,
        n_repeat: int,
        orig_kv_heads: int,
        new_kv_heads: int,
        num_attention_heads: int,
        hidden_size: int,
    ) -> nn.Module:
        """
        Mutates one attention module in-place by replicating its KV projection weights.
        """
        head_dim = getattr(original_module, "head_dim", hidden_size // num_attention_heads)
        k_proj = get_projection_layer(original_module, ("k_proj", "key_proj"))
        v_proj = get_projection_layer(original_module, ("v_proj", "value_proj"))
        duplicate_kv_projection_weights(
            k_proj,
            orig_kv_heads,
            n_repeat,
            head_dim,
            hidden_size,
            layer_name=f"{original_module.__class__.__name__}.k_proj",
        )
        duplicate_kv_projection_weights(
            v_proj,
            orig_kv_heads,
            n_repeat,
            head_dim,
            hidden_size,
            layer_name=f"{original_module.__class__.__name__}.v_proj",
        )

        if hasattr(original_module, "num_key_value_heads"):
            original_module.num_key_value_heads = new_kv_heads
        if hasattr(original_module, "n_kv_heads"):
            original_module.n_kv_heads = new_kv_heads
        n_kv_groups = num_attention_heads // new_kv_heads
        if hasattr(original_module, "num_key_value_groups"):
            original_module.num_key_value_groups = n_kv_groups
        if hasattr(original_module, "n_kv_groups"):
            original_module.n_kv_groups = n_kv_groups
        return original_module

    @classmethod
    def apply(cls, model: nn.Module, num_replicate_kv_heads: int | None = None, **kwargs) -> tuple[nn.Module, bool]:
        """
        Replicates KV heads in attention modules based on provided multiplier.

        Args:
            model: The model to apply the transform to.
            kwargs: Additional arguments for the transformation.
            num_replicate_kv_heads: The number of times to repeat the KV heads.
        """
        if num_replicate_kv_heads is None:
            n_repeat = kwargs.pop("num_replicate_kv_heads", 1)
        else:
            kwargs.pop("num_replicate_kv_heads", None)
            n_repeat = num_replicate_kv_heads
        if n_repeat is None or n_repeat <= 1:
            return model, False

        text_model = get_text_model(model)
        cfg = text_model.config
        if is_mla_model(text_model):
            logger.warning("Skipping RepeatKVTransform: MLA models don't apply replicate KV changes.")
            return model, False

        current_kv_heads = resolve_kv_heads(cfg)
        configured_orig_kv_heads = getattr(cfg, "orig_kv_heads", None)
        if configured_orig_kv_heads is not None:
            if configured_orig_kv_heads < 1:
                raise ValueError(f"Invalid stored orig_kv_heads in config for RepeatKV: {configured_orig_kv_heads}")
            expected_kv_heads = configured_orig_kv_heads * n_repeat
            if current_kv_heads == expected_kv_heads:
                logger.warning("Skipping RepeatKVTransform: model already has requested repeated KV heads.")
                return model, False
            orig_kv_heads = configured_orig_kv_heads
        else:
            orig_kv_heads = current_kv_heads
        num_attention_heads = resolve_attention_heads(cfg)
        hidden_size = resolve_hidden_size(cfg)

        if orig_kv_heads is None or num_attention_heads is None or hidden_size is None:
            raise ValueError(
                "Unable to resolve attention/KV heads or hidden size from config for RepeatKV transform. "
                f"Supported attention keys={ATTENTION_HEAD_CONFIG_KEYS}, kv keys={KV_HEAD_CONFIG_KEYS}, "
                f"hidden size keys={HIDDEN_SIZE_CONFIG_KEYS}."
            )
        if orig_kv_heads < 1 or num_attention_heads < 1:
            raise ValueError(
                f"Invalid head values for RepeatKV transform: "
                f"num_attention_heads={num_attention_heads}, num_key_value_heads={orig_kv_heads}"
            )
        if orig_kv_heads == num_attention_heads:
            raise ValueError(
                "RepeatKV is supported only for GQA/MQA models where num_key_value_heads is smaller than "
                f"num_attention_heads; got num_attention_heads={num_attention_heads}, "
                f"num_key_value_heads={orig_kv_heads}."
            )
        new_kv_heads = n_repeat * orig_kv_heads
        if new_kv_heads > num_attention_heads or (num_attention_heads % new_kv_heads) != 0:
            raise ValueError(
                f"Invalid RepeatKV configuration: num_attention_heads={num_attention_heads}, "
                f"orig_kv_heads={orig_kv_heads}, num_replicate_kv_heads={n_repeat}, new_kv_heads={new_kv_heads}. "
                "Expected new_kv_heads <= num_attention_heads and divisibility."
            )

        logger.warning(f"Original KV heads: {orig_kv_heads}")
        logger.warning(f"Modified KV heads: {new_kv_heads}")
        for block in text_model.layers:
            try:
                attn = get_attention_module(block)
            except AttributeError:
                logger.warning(
                    f"Skipping RepeatKV for block type {block.__class__.__name__}: "
                    "no replicable attention module found."
                )
                continue
            cls.mutate(
                attn,
                block,
                n_repeat,
                orig_kv_heads,
                new_kv_heads,
                num_attention_heads,
                hidden_size,
            )
        cfg.orig_kv_heads = orig_kv_heads
        set_kv_head_aliases(cfg, new_kv_heads)
        return model, True


class SpDTransform:
    """
    Apply generic QEffForCausalLM forward pass to extract `num_speculative_tokens+1` hidden states before computing logits during decode phase and extract last predicted token during prefill.
    This is only needed if user is exporting Target Language Model (TLM) for Speculative Decoding to validate output logits
    against the speculated tokens from a smaller model.
    Other than the computed logits, there should be no difference between the SpD Transformed model and its corresponding cunterpart.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        # Llama
        QEffLlamaForCausalLM,
        QEffQwen2ForCausalLM,
        QEffQwen3ForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: dict | None = None, **kwargs) -> tuple[nn.Module, bool]:
        transformed = False
        pretrained_model_name_or_path_temp = kwargs.pop("pretrained_model_name_or_path", None)
        if qaic_config is None or (speculative_model_type := qaic_config.get("speculative_model_type")) is None:
            return model, transformed
        elif speculative_model_type not in (
            supported_spd_model_types := [SPD_TARGET] + list(model_type_registry.keys())
        ):
            raise ValueError(
                f"Speculative model type {speculative_model_type} is not supported. we currently only support {supported_spd_model_types}"
            )
        elif (model_class := model.__class__) in cls._module_mapping:
            model.forward = MethodType(tlm_forward, model)
            if speculative_model_type != SPD_TARGET:
                # build and attach draft mlp
                pretrained_model_name_or_path = qaic_config["pretrained_model_name_or_path"]
                model = build_and_attach_mlp(
                    model, pretrained_model_name_or_path, speculative_model_type=speculative_model_type, **kwargs
                )
            transformed = True
        else:
            raise NotImplementedError(
                f"model class {model_class} does not yet support returning multiple logits to keep."
            )
        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path_temp
        return model, transformed


class SamplerTransform:
    """
    Add nodes at the output of any generic QEffForCausalLM model to enable the
    sampling of next tokens at the device (instead of the host) and return the
    next tokens and/or probability distributions.

    Note: To achieve this, the generic QEffForCausalLM model must provide the
    logits as output.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        QEffFalconForCausalLM,
        QEffGemmaForCausalLM,
        QEffGemma3DecoderWrapper,
        QEffGPT2LMHeadModel,
        QEffGPTJForCausalLM,
        QEffGraniteForCausalLM,
        QEffGraniteMoeForCausalLM,
        QEffInternDecoderWrapper,
        QEffLlamaForCausalLM,
        QEffLlama4DecoderWrapper,
        QEFFLlavaDecoderWrapper,
        QEffLlavaNextDecoderWrapper,
        QEffMptForCausalLM,
        QEffPhi3ForCausalLM,
        QEffQwen2ForCausalLM,
        QEffQwen_2_5_vl_DecoderWrapper,
        QEffQwen3ForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: dict | None = None, **kwargs) -> tuple[nn.Module, bool]:
        transformed = False
        if qaic_config is None or not qaic_config.get("include_sampler", False):
            return model, transformed
        elif (model_class := model.__class__) in cls._module_mapping:
            model.old_forward = model.forward
            model.forward = MethodType(sampler_forward, model)
            transformed = True
        else:
            raise NotImplementedError(f"Model class {model_class} does not support on device sampling.")
        return model, transformed


class DFlashTransform(ModuleMappingTransform):
    """Replace QEff Qwen3 modules with DFlash variants when dflash_dlm is set."""

    _module_mapping = {
        QEffQwen3Attention: QEffDFlashAttention,
        QEffQwen3DecoderLayer: QEffDFlashDecoderLayer,
        QEffQwen3Model: QEffDFlashModel,
        QEffQwen3ForCausalLM: QEffDFlashForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: dict | None = None, **kwargs) -> tuple[nn.Module, bool]:
        if not (qaic_config and qaic_config.get("dflash_dlm", False)):
            return model, False
        if type(model) is not QEffQwen3ForCausalLM:
            raise NotImplementedError(
                f"DFlash DLM does not support model class {type(model).__name__}. "
                "Supported model class: QEffQwen3ForCausalLM."
            )
        return super().apply(model)


class DFlashDLMTransform:
    """Inject lm_head/embed_tokens from the TLM checkpoint (dflash_tlm_repo) and drop fc/hidden_norm."""

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: dict | None = None, **kwargs) -> tuple[nn.Module, bool]:
        if not (qaic_config and qaic_config.get("dflash_dlm", False)):
            return model, False

        inner = model.model
        for attr in ("fc", "hidden_norm"):
            if hasattr(inner, attr):
                delattr(inner, attr)

        tlm_repo = qaic_config.get("dflash_tlm_repo")
        if not tlm_repo:
            return model, True

        w = load_checkpoint_weights(tlm_repo, {"lm_head.weight", "lm_head.bias", "model.embed_tokens.weight"})
        embed_w = w.get("model.embed_tokens.weight")
        lm_head_w = w.get("lm_head.weight", embed_w)  # tie_word_embeddings: lm_head is a view of embed_tokens
        if lm_head_w is None or embed_w is None:
            return model, True

        with torch.no_grad():
            model.lm_head.weight.data.copy_(lm_head_w.float())
            inner.embed_tokens.weight.data.copy_(embed_w.float())
            if w.get("lm_head.bias") is not None:
                model.lm_head.bias = nn.Parameter(w["lm_head.bias"].float())
        return model, True


class DFlashTLMTransform:
    """Attach fc/hidden_norm (weights from dflash_dlm_repo, fc scaled for fp16 range) and set target_layer_ids."""

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: dict | None = None, **kwargs) -> tuple[nn.Module, bool]:
        target_layer_ids = qaic_config.get("target_layer_ids") if qaic_config else None
        if not target_layer_ids:
            return model, False

        # Conditional-generation (VLM) models (e.g. gemma4) nest the language config
        # under text_config and the decoder under model.model.language_model; plain
        # CausalLM models (qwen3/llama) expose them directly.
        config = model.config
        text_config = getattr(config, "text_config", config)
        hidden_size = text_config.hidden_size
        inner = model.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        n = len(target_layer_ids)
        # Keep injected modules aligned with the decoder hidden-state precision.
        reference_weight = inner.embed_tokens.weight
        module_device = reference_weight.device
        module_dtype = reference_weight.dtype

        # Skip if a caller pre-injected fc/hidden_norm before constructing the model.
        if not (hasattr(inner, "fc") and hasattr(inner, "hidden_norm")):
            model_type = getattr(config, "model_type", "")
            eps = getattr(text_config, "rms_norm_eps", getattr(config, "rms_norm_eps", 1e-6))
            if "gemma" in model_type:
                RMSNorm = Gemma4RMSNorm
            elif "qwen3_vl" in model_type:
                RMSNorm = Qwen3VLTextRMSNorm
            elif "qwen3" in model_type:
                RMSNorm = Qwen3RMSNorm
            elif "llama" in model_type:
                RMSNorm = LlamaRMSNorm
            else:
                RMSNorm = nn.RMSNorm
            inner.fc = nn.Linear(n * hidden_size, hidden_size, bias=False).to(device=module_device, dtype=module_dtype)
            inner.hidden_norm = RMSNorm(hidden_size, eps=eps).to(device=module_device, dtype=module_dtype)

            dlm_repo = qaic_config.get("dflash_dlm_repo")
            w = load_checkpoint_weights(dlm_repo, {"fc.weight", "hidden_norm.weight"}) if dlm_repo else {}
            if "fc.weight" in w and "hidden_norm.weight" in w:
                inner.fc.weight.data.copy_(w["fc.weight"].to(device=module_device, dtype=module_dtype))
                inner.hidden_norm.weight.data.copy_(
                    w["hidden_norm.weight"].to(device=module_device, dtype=module_dtype)
                )
                # RMSNorm(x/s) == RMSNorm(x): scale fc down to keep activations in fp16 range.
                with torch.no_grad():
                    bound = (inner.fc.in_features**0.5) * inner.fc.weight.data.float().norm(dim=1).max().item()
                    inner.fc.weight.data.div_(max(bound / _DFLASH_TARGET_ABSMAX, 1.0))
            else:
                warnings.warn(f"DFlashTLMTransform: fc/hidden_norm not found in {dlm_repo!r}; left random.")

        inner.fc.to(device=module_device, dtype=module_dtype)
        inner.hidden_norm.to(device=module_device, dtype=module_dtype)
        inner.target_layer_ids = target_layer_ids
        return model, True


class VlmKVOffloadTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        # Llama
        MllamaTextCrossAttention: QEffMllamaTextCrossAttentionTwoQPC,
    }


class VlmNoKVOffloadTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        # Llama
        MllamaTextCrossAttention: QEffMllamaTextCrossAttentionSingleQPC,
    }


class KVCacheExternalModuleMapperTransform(ExternalModuleMapperTransform):
    _match_class_replace_method = {}
    _match_string_replace_method = {
        "InternVLChatModel": {
            "forward": QEffInternVLModel.forward,
            "get_dummy_inputs": QEffInternVLModel.get_dummy_inputs,
            "get_specializations": QEffInternVLModel.get_specializations,
            "get_onnx_dynamic_axes": QEffInternVLModel.get_onnx_dynamic_axes,
            "get_output_names": QEffInternVLModel.get_output_names,
            "get_inputs_info": QEffInternVLModel.get_inputs_info,
            "get_qeff_vision_encoder": QEffInternVLModel.get_qeff_vision_encoder,
            "get_qeff_language_decoder": QEffInternVLModel.get_qeff_language_decoder,
        },
        "InternVisionEmbeddings": {"forward": QEffInternVisionEmbeddings.forward},
        # Mapping for Molmo
        "MolmoForCausalLM": {
            "forward": QEffMolmoModel.forward,
            "get_qeff_vision_encoder": QEffMolmoModel.get_qeff_vision_encoder,
            "get_qeff_language_decoder": QEffMolmoModel.get_qeff_language_decoder,
            "get_specializations": QEffMolmoModel.get_specializations,
            "get_onnx_dynamic_axes": QEffMolmoModel.get_onnx_dynamic_axes,
            "get_output_names": QEffMolmoModel.get_output_names,
            "get_dummy_inputs": QEffMolmoModel.get_dummy_inputs,
            "get_inputs_info": QEffMolmoModel.get_inputs_info,
        },
        "RMSLayerNorm": {"forward": CustomRMSNormAIC.forward},
        # "MolmoForCausalLM": {"forward": QEffMolmoForCausalLM.forward},
        "Molmo": {"forward": QEffMolmo.forward},
        "MolmoSequentialBlock": {
            "forward": QEffMolmoSequentialBlock.forward,
            "attention": QEffMolmoBlock.attention,
            "__qeff_init__": QEffMolmoBlock.__qeff_init__,
        },
        "MolmoBlock": {
            "attention": QEffMolmoBlock.attention,
            "__qeff_init__": QEffMolmoBlock.__qeff_init__,
        },
        "MultiHeadDotProductAttention": {
            "forward": QEffMultiHeadDotProductAttention.forward,
        },
        # Mapping for grok1 model
        "Grok1ModelForCausalLM": {"forward": QEffGrok1ModelForCausalLM.forward},
        "Grok1Model": {
            "forward": QEffGrok1Model.forward,
            "__qeff_init__": QEffGrok1Model.__qeff_init__,
        },
        "DecoderLayer": {
            "forward": QEffGrok1DecoderLayer.forward,
            "__qeff_init__": QEffGrok1DecoderLayer.__qeff_init__,
        },
        "MultiHeadAttention": {
            "forward": QEffGrok1MultiHeadAttention.forward,
        },
        "RMSNorm": {
            "forward": QEFFGrok1CustomRMSNormAIC.forward,
        },
        "KimiK25ForConditionalGeneration": {
            "_qeff_merge_input_ids_with_image_features": QEffKimiK25ForConditionalGeneration._qeff_merge_input_ids_with_image_features,
            "get_qeff_vision_encoder": QEffKimiK25ForConditionalGeneration.get_qeff_vision_encoder,
            "get_qeff_language_decoder": QEffKimiK25ForConditionalGeneration.get_qeff_language_decoder,
            "get_specializations": QEffKimiK25ForConditionalGeneration.get_specializations,
            "get_onnx_dynamic_axes": QEffKimiK25ForConditionalGeneration.get_onnx_dynamic_axes,
            "get_output_names": QEffKimiK25ForConditionalGeneration.get_output_names,
            "get_dummy_inputs": QEffKimiK25ForConditionalGeneration.get_dummy_inputs,
        },
        "MoonViT3dEncoder": {
            "__qeff_init__": QEffMoonViT3dEncoder.__qeff_init__,
        },
        "Learnable2DInterpPosEmbDivided_fixed": {
            "__qeff_init__": QEffLearnable2DInterpPosEmbDivided_fixed.__qeff_init__,
        },
        "DeepseekV3ForCausalLM": {
            "forward": QEffDeepseekV3ForCausalLM.forward,
            "get_submodules_for_export": QEffDeepseekV3ForCausalLM.get_submodules_for_export,
            "get_dummy_pkv_cache": QEffDeepseekV3ForCausalLM.get_dummy_pkv_cache,
        },
        "DeepseekV3Model": {"forward": QEffDeepseekV3Model.forward, "__qeff_init__": QEffDeepseekV3Model.__qeff_init__},
        "DeepseekV3DecoderLayer": {
            "forward": QEffDeepseekV3DecoderLayer.forward,
        },
        "DeepseekV3Attention": {
            "forward": QEffDeepseekV3Attention.forward,
            "forward_full_kv": QEffDeepseekV3Attention.forward_full_kv,
            "forward_full_kv_h_blocking": QEffDeepseekV3Attention.forward_full_kv_h_blocking,
            "fused_forward": QEffDeepseekV3Attention.fused_forward,
            "fused_forward_h_blocking": QEffDeepseekV3Attention.fused_forward_h_blocking,
            "fused_forward_kv_blocking": QEffDeepseekV3Attention.fused_forward_kv_blocking,
            "fused_forward_orig": QEffDeepseekV3Attention.fused_forward_orig,
            "__qeff_init__": QEffDeepseekV3Attention.__qeff_init__,
        },
        "DeepseekV3RMSNorm": {
            "forward": QEffDeepseekV3CustomRMSNormAIC.forward,
        },
    }


class PrefillOnlyExternalModuleMapperTransform(ExternalModuleMapperTransform):
    _match_class_replace_method = {}
    _match_string_replace_method = {}


class RevertPrefillOnlyExternalModuleMapperTransform(ExternalModuleMapperTransform):
    _match_class_replace_method = {}
    _match_string_replace_method = {}


class T5ModelTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        T5Attention: QEffT5Attention,
        T5LayerNorm: QEffT5LayerNorm,
        T5Stack: QEffT5Stack,
    }


class TextClassificationTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        # DebertaV2
        DisentangledSelfAttention: QEffDisentangledSelfAttention,
    }


class PoolingTransform:
    """
    Apply a pooling transformation to the model. This transformation appends a pooling layer to the model, allowing for the reduction of spatial dimensions in the output.
    The pooling layer can be configured to use different pooling methods, such as max pooling or average pooling.
    """

    @classmethod
    def apply(cls, model: nn.Module, pooling: str | Callable) -> tuple[nn.Module, bool]:
        transformed = False
        pooling_method = (
            POOLING_MAP[pooling]
            if isinstance(pooling, str) and pooling in POOLING_MAP
            else validate_user_pooling_function(pooling)
        )
        model = PooledModel(model, pooling_method)
        warnings.warn("Pooling is applied to the model.")
        return model, transformed


def get_decoder_layer_classes_for_export(model: nn.Module) -> set:
    """
    Dynamically determine which DecoderLayer classes should be exported as functions
    based on the model's architecture using the existing KVCacheTransform mapping.
    """
    # Define patterns that identify decoder layer classes
    DECODER_LAYER_PATTERNS = ["DecoderLayer", "Block", "Layer"]

    # Get all QEff classes that are decoder layers from the existing mapping
    decoder_layer_classes = set()

    for qeff_class in KVCacheTransform._module_mapping.values():
        # Check if the QEff class name contains decoder layer patterns
        qeff_class_name = qeff_class.__name__
        if any(pattern in qeff_class_name for pattern in DECODER_LAYER_PATTERNS):
            decoder_layer_classes.add(qeff_class)

    # Filter to only include classes that are actually used in the current model
    model_decoder_classes = set()
    for module in model.modules():
        if module.__class__ in decoder_layer_classes:
            model_decoder_classes.add(module.__class__)

    return model_decoder_classes


class BlockingAttentionTransform:
    _skip_classes = {}

    @classmethod
    def apply(cls, model: nn.Module, attn_blocking_config) -> tuple[nn.Module, bool]:
        transformed = False
        model_config = getattr(model, "config", None) or getattr(getattr(model, "model", None), "config", None)
        model_architectures = getattr(model_config, "architectures", None) or []
        supported_attention_classes = {
            qeff_class
            for qeff_class in KVCacheTransform._module_mapping.values()
            if qeff_class.__name__.endswith("Attention")
        }
        for module in model.modules():
            if type(module) in cls._skip_classes:
                warnings.warn(f"Blocking is not yet supported for {type(module)}.")
                continue
            if type(module) in supported_attention_classes or "DeepseekV3ForCausalLM" in model_architectures:
                module.attn_blocking_config = attn_blocking_config
                transformed = True
            elif module.__class__.__name__.endswith("Attention") and type(module) not in supported_attention_classes:
                warnings.warn(f"Blocking is not yet supported for {type(module)}.")
        return model, transformed


class GatedDeltaConfigTransform:
    @classmethod
    def apply(cls, model: nn.Module, gated_delta_config: dict | None = None) -> tuple[nn.Module, bool]:
        gated_delta_config = gated_delta_config or {}
        full_state_update = bool(gated_delta_config.get("full_state_update", False))
        target_modules = [module for module in model.modules() if hasattr(module, "torch_chunk_gated_delta_rule_qeff")]
        full_state_targets = [module for module in target_modules if hasattr(module, "gdn_full_state_update")]
        if full_state_update and not full_state_targets:
            raise ValueError("qaic_config['gdn_full_state_update'] is not supported for this model")
        if not target_modules:
            return model, False

        if not gated_delta_config:
            # Preserve the legacy chunk-size reset and independently clear the
            # model-specific full-state mode between prefill/decode compiles.
            transformed = False
            for module in target_modules:
                if getattr(module, "gdn_chunk_size", 64) != 64:
                    module.gdn_chunk_size = 64
                    if hasattr(module, "__qeff_init__"):
                        module.__qeff_init__()
                    transformed = True
            for module in full_state_targets:
                if bool(module.gdn_full_state_update):
                    module.gdn_full_state_update = False
                    transformed = True
            return model, transformed

        chunk_size = gated_delta_config.get("chunk_size")
        chunk_size = int(chunk_size) if chunk_size is not None else None

        transformed = False
        for module in target_modules:
            if chunk_size is not None and chunk_size > 0:
                module.gdn_chunk_size = chunk_size
                if hasattr(module, "__qeff_init__"):
                    module.__qeff_init__()
                transformed = True
        for module in full_state_targets:
            if full_state_update or bool(module.gdn_full_state_update) != full_state_update:
                transformed = True
            module.gdn_full_state_update = full_state_update

        return model, transformed


def _iter_optimized_moe_modules(model: nn.Module):
    from QEfficient.transformers.moe import QEffMoEBlockMixin

    for module in model.modules():
        has_structural_contract = callable(getattr(module, "route", None)) and callable(
            getattr(module, "transform_weights", None)
        )
        if isinstance(module, QEffMoEBlockMixin) or has_structural_contract:
            yield module


def _get_moe_num_experts(module: nn.Module) -> int | None:
    weights = getattr(module, "moe_weights", None)
    if weights is not None:
        return int(weights.num_experts)
    quantized_gate_weights = getattr(module, "all_gate_qweight", None)
    if quantized_gate_weights is not None:
        return int(quantized_gate_weights.shape[0])
    num_experts = getattr(module, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    experts = getattr(module, "experts", None)
    num_experts = getattr(experts, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    gate = getattr(module, "gate", None)
    num_experts = getattr(gate, "num_experts", None)
    return int(num_experts) if num_experts is not None else None


def _resolve_expert_parallel_layout(
    *,
    num_experts: int,
    num_devices: int,
    num_cores: int,
    cores_per_expert: int,
) -> tuple[int, int, int, int | None]:
    if num_devices <= 0:
        raise ValueError("num_devices must be greater than zero for MoE expert parallelism")
    if num_cores <= 0:
        raise ValueError("num_cores must be greater than zero for MoE expert parallelism")
    if cores_per_expert <= 0:
        raise ValueError("cores_per_expert must be greater than zero for MoE expert parallelism")

    required_expert_cores = num_experts * cores_per_expert
    total_avl_cores = num_devices * num_cores
    if required_expert_cores < total_avl_cores and total_avl_cores % required_expert_cores == 0:
        num_pipeline_stages = 1
    elif required_expert_cores % total_avl_cores == 0:
        num_pipeline_stages = required_expert_cores // total_avl_cores
    else:
        raise ValueError(
            "num_experts * cores_per_expert "
            "(num_experts * cores_per_expert) must be divisible by total_avl_cores (num_devices * num_cores)"
        )
    if num_pipeline_stages <= 0:
        raise ValueError(f"num_pipeline_stages ({num_pipeline_stages}) must be greater than zero")
    if num_experts % num_pipeline_stages != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by num_pipeline_stages ({num_pipeline_stages})"
        )
    num_parallelized_experts = num_experts // num_pipeline_stages
    if num_parallelized_experts % num_devices != 0:
        raise ValueError(
            f"num_parallelized_experts ({num_parallelized_experts}) must be divisible by num_devices ({num_devices})"
        )
    if num_devices > 1 and num_cores % cores_per_expert != 0:
        raise ValueError(f"num_cores ({num_cores}) must be divisible by cores_per_expert ({cores_per_expert})")
    experts_per_soc = num_cores // cores_per_expert if num_devices > 1 else None
    return total_avl_cores, num_pipeline_stages, num_parallelized_experts, experts_per_soc


class OptimizedMoEMapperTransform(ModuleMappingTransform):
    """Replace in-tree MoE components with QEff implementations."""

    _module_mapping = {
        # GLM4-MoE
        Glm4MoeMoE: QEffGlm4MoeMoE,
        Glm4MoeTopkRouter: QEffGlm4MoeTopkRouter,
        # Llama4
        Llama4TextMoe: QEffLlama4TextMoe,
        Llama4TextExperts: QEffLlama4TextExperts,
        Llama4Router: QEffLlama4Router,
        # Qwen3-MoE
        Qwen3MoeExperts: QEffQwen3MoeExperts,
        Qwen3MoeSparseMoeBlock: QEffQwen3MoeSparseMoeBlock,
        Qwen3MoeTopKRouter: QEffQwen3MoeTopKRouter,
        # Qwen3-VL-MoE
        Qwen3VLMoeTextExperts: QEffQwen3VLMoeTextExperts,
        Qwen3VLMoeTextSparseMoeBlock: QEffQwen3VLMoeTextSparseMoeBlock,
        Qwen3VLMoeTextTopKRouter: QEffQwen3VLMoeTextTopKRouter,
        # Qwen3.5-MoE
        Qwen3_5MoeExperts: QEffQwen3_5MoeExperts,
        Qwen3_5MoeSparseMoeBlock: QEffQwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeTopKRouter: QEffQwen3_5MoeTopKRouter,
        # Gemma4
        Gemma4TextRouter: QEffGemma4TextRouter,
        # GPT-OSS
        GptOssMLP: QEffGptOssMLP,
        GptOssExperts: QEffGptOssExperts,
        # GraniteMoE
        GraniteMoeMoE: QEffGraniteMoeMoE,
        GraniteMoeParallelExperts: QEffGraniteMoeParallelExperts,
        GraniteMoeTopKGating: QEffGraniteMoeTopKGating,
        # Mixtral
        MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        model, mapped = super().apply(model)
        return model, mapped or any(True for _ in _iter_optimized_moe_modules(model))


class ExternalOptimizedMoEMapperTransform(ExternalModuleMapperTransform):
    """Bind optimized MoE methods for external model code."""

    _match_class_replace_method = {}
    _match_string_replace_method = {
        "MoeBlock": {
            "forward": QEffGrok1MoeBlock.forward,
            "get_supported_moe_flavours": QEffMoEBlockMixin.get_supported_moe_flavours,
            "routing_input": QEffMoEBlockMixin.routing_input,
            "execute_moe_flavour": QEffMoEBlockMixin.execute_moe_flavour,
            "moe_dispatch": QEffMoEBlockMixin.moe_dispatch,
            "transform_weights": QEffGrok1MoeBlock.transform_weights,
            "route": QEffGrok1MoeBlock.route,
            "moe_profile": QEffGrok1MoeBlock.moe_profile,
            "apply_shared_experts": QEffMoEBlockMixin.apply_shared_experts,
            "_moe_return_router_logits": True,
            "_moe_flavour": MoEFlavour.DECODE_BMM,
            "supported_moe_flavours": QEffGrok1MoeBlock.supported_moe_flavours,
            "supports_moe_decode_bmm": QEffGrok1MoeBlock.supports_moe_decode_bmm,
            "__qeff_init__": QEffGrok1MoeBlock.__qeff_init__,
        },
        "DeepseekV3MoE": {
            "forward": QEffDeepseekV3MoE.forward,
            "get_supported_moe_flavours": QEffDeepseekV3MoE.get_supported_moe_flavours,
            "routing_input": QEffDeepseekV3MoE.routing_input,
            "execute_moe_flavour": QEffMoEBlockMixin.execute_moe_flavour,
            "moe_dispatch": QEffMoEBlockMixin.moe_dispatch,
            "transform_weights": QEffDeepseekV3MoE.transform_weights,
            "_stack_quantized_projection_params": QEffDeepseekV3MoE._stack_quantized_projection_params,
            "_transform_quantized_expert_weights": QEffDeepseekV3MoE._transform_quantized_expert_weights,
            "route": QEffDeepseekV3MoE.route,
            "moe_profile": QEffDeepseekV3MoE.moe_profile,
            "apply_shared_experts": QEffDeepseekV3MoE.apply_shared_experts,
            "_moe_return_router_logits": False,
            "_moe_flavour": MoEFlavour.DECODE_BMM,
            "supported_moe_flavours": QEffDeepseekV3MoE.supported_moe_flavours,
            "supports_moe_decode_bmm": QEffDeepseekV3MoE.supports_moe_decode_bmm,
            "__qeff_init__": QEffDeepseekV3MoE.__qeff_init__,
        },
    }


class OptimizedMoEWeightsTransform(PytorchTransform):
    """Canonicalize MoE expert weights for modules using shared MoE flavours."""

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        transformed = False
        for module in list(_iter_optimized_moe_modules(model)):
            if getattr(module, "weights_transformed", False):
                continue
            module.transform_weights()
            gc.collect()
            transformed = True
        return model, transformed


class OptimizedMoEExportConfigTransform(PytorchTransform):
    """Assign MoE export flavour, blocking attributes, and hash parameters."""

    @classmethod
    def apply(
        cls,
        model: nn.Module,
        *,
        prefill_only: bool = False,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = DEFAULT_AIC_NUM_CORES,
        qaic_config: dict | None = None,
        prefill_seq_len: int | None = None,
        hash_params: dict | None = None,
    ) -> tuple[nn.Module, bool]:
        from QEfficient.transformers.moe import MoEFlavour, select_moe_flavour

        moe_config = (qaic_config or {}).get("moe_config", {}) or {}
        requested_flavour = moe_config.get("flavour")

        if "num_nsp" in moe_config:
            logger.warning("qaic_config['moe_config']['num_nsp'] is deprecated and ignored for MoE expert parallelism.")
        cores_per_expert = int(moe_config.get("cores_per_expert", 1))
        tree_reduce = moe_config.get("tree_reduce", False)
        if not isinstance(tree_reduce, bool):
            raise TypeError("qaic_config['moe_config']['tree_reduce'] must be a boolean")
        chunk_size_key = (
            "expert_parallel_chunk_size" if "expert_parallel_chunk_size" in moe_config else "packed_chunk_size"
        )
        expert_parallel_chunk_size_requested = (
            chunk_size_key in moe_config and moe_config.get(chunk_size_key) is not None
        )
        expert_parallel_chunk_size = moe_config.get(chunk_size_key, MOE_PREFILL_PACKED_CHUNK_SIZE)
        if expert_parallel_chunk_size is None:
            expert_parallel_chunk_size = MOE_PREFILL_PACKED_CHUNK_SIZE
        expert_parallel_chunk_size = int(expert_parallel_chunk_size)
        if expert_parallel_chunk_size <= 0:
            raise ValueError("moe expert_parallel_chunk_size must be greater than zero")
        compile_seq_len = prefill_seq_len or ONNX_EXPORT_EXAMPLE_SEQ_LEN
        compile_batch_size = int(batch_size or 1)
        if compile_batch_size <= 0:
            raise ValueError("batch_size must be greater than zero for MoE expert parallelism")
        compile_num_tokens = compile_batch_size * compile_seq_len
        num_packed_chunks = max(1, -(-compile_num_tokens // expert_parallel_chunk_size))

        transformed = False
        flavour = None
        uses_expert_parallel = False
        layout = None
        for module in list(_iter_optimized_moe_modules(model)):
            get_supported_moe_flavours = getattr(module, "get_supported_moe_flavours", None)
            if callable(get_supported_moe_flavours):
                supported_flavours = get_supported_moe_flavours()
            else:
                supported_flavours = QEffMoEBlockMixin.get_supported_moe_flavours(module)
            flavour = select_moe_flavour(
                supported_flavours=supported_flavours,
                is_prefill=prefill_only,
                requested_flavour=requested_flavour,
            )
            module._moe_flavour = flavour
            if flavour is MoEFlavour.EXPERT_PARALLEL:
                uses_expert_parallel = True
                num_experts = _get_moe_num_experts(module)
                if num_experts is None:
                    raise AttributeError(f"{type(module).__name__} does not expose num_experts for expert parallelism")
                layout = _resolve_expert_parallel_layout(
                    num_experts=num_experts,
                    num_devices=int(num_devices),
                    num_cores=int(num_cores),
                    cores_per_expert=cores_per_expert,
                )
                total_avl_cores, num_pipeline_stages, num_parallelized_experts, experts_per_soc = layout
                module.num_experts = num_experts
                module.num_devices = int(num_devices)
                module.cores_per_expert = cores_per_expert
                module.total_avl_cores = total_avl_cores
                module.num_pipeline_stages = num_pipeline_stages
                module.num_parallelized_experts = num_parallelized_experts
                module.experts_per_soc = experts_per_soc
                module.tree_reduce = tree_reduce
                module.expert_parallel_num_packed_chunks = num_packed_chunks
                module.expert_blocking_packed_chunk_size = expert_parallel_chunk_size
            transformed = True

        if uses_expert_parallel and compile_num_tokens % expert_parallel_chunk_size != 0:
            logger.warning(
                f"qaic_config['moe_config']['expert_parallel_chunk_size']={expert_parallel_chunk_size} does not evenly divide "
                f"the compile token count {compile_num_tokens}; the number of packed chunks will be {num_packed_chunks}."
            )

        if transformed and expert_parallel_chunk_size_requested and not uses_expert_parallel:
            logger.warning(
                "qaic_config['moe_config']['expert_parallel_chunk_size'] is only used for "
                "moe flavour 'expert_parallel'; the provided value will be ignored."
            )
        if (
            transformed
            and ("cores_per_expert" in moe_config or "tree_reduce" in moe_config)
            and not uses_expert_parallel
        ):
            logger.warning(
                "qaic_config['moe_config']['cores_per_expert'] and "
                "qaic_config['moe_config']['tree_reduce'] are only used for moe flavour 'expert_parallel'."
            )

        if transformed and hash_params is not None:
            hash_params["moe_prefill_flavour"] = (flavour or MoEFlavour.DECODE_BMM).value
            if flavour is MoEFlavour.EXPERT_PARALLEL and layout is not None:
                total_avl_cores, num_pipeline_stages, num_parallelized_experts, _ = layout
                hash_params["moe_prefill_total_avl_cores"] = total_avl_cores
                hash_params["moe_prefill_cores_per_expert"] = cores_per_expert
                hash_params["moe_prefill_tree_reduce"] = tree_reduce
                hash_params["moe_prefill_num_pipeline_stages"] = num_pipeline_stages
                hash_params["moe_prefill_num_parallelized_experts"] = num_parallelized_experts
                hash_params["moe_prefill_expert_parallel_chunk_size"] = expert_parallel_chunk_size
                hash_params["moe_prefill_num_packed_chunks"] = num_packed_chunks
            else:
                hash_params.pop("moe_prefill_num_nsp", None)
                hash_params.pop("moe_prefill_packed_chunk_size", None)
                hash_params.pop("moe_prefill_total_avl_cores", None)
                hash_params.pop("moe_prefill_cores_per_expert", None)
                hash_params.pop("moe_prefill_tree_reduce", None)
                hash_params.pop("moe_prefill_num_pipeline_stages", None)
                hash_params.pop("moe_prefill_num_parallelized_experts", None)
                hash_params.pop("moe_prefill_expert_parallel_chunk_size", None)
                hash_params.pop("moe_prefill_num_packed_chunks", None)

        return model, transformed


def _replace_moe_weight_aliases(model: nn.Module, old_weights: nn.Module, new_weights: nn.Module) -> None:
    for module in list(model.modules()):
        for name, child in list(module._modules.items()):
            if child is old_weights:
                setattr(module, name, new_weights)


class OptimizedMoEExpertParallelWeightsTransform(PytorchTransform):
    """Pack or restore MoE weights according to the selected MoE export flavour."""

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        transformed = False
        for module in list(_iter_optimized_moe_modules(model)):
            if not getattr(module, "weights_transformed", False):
                continue
            weights = getattr(module, "moe_weights", None)
            if weights is None:
                continue

            flavour = getattr(module, "_moe_flavour", MoEFlavour.DECODE_BMM)
            if not isinstance(flavour, MoEFlavour):
                flavour = MoEFlavour(flavour)

            if flavour is MoEFlavour.EXPERT_PARALLEL:
                num_pipeline_stages = getattr(module, "num_pipeline_stages", None)
                num_parallelized_experts = getattr(module, "num_parallelized_experts", None)
                if num_pipeline_stages is None or num_parallelized_experts is None:
                    num_pipeline_stages = weights.gate.shape[1] if weights.gate.ndim == 4 else weights.num_experts
                    num_parallelized_experts = weights.gate.shape[0] if weights.gate.ndim == 4 else 1
                new_weights = pack_moe_weights_for_expert_parallel(
                    weights,
                    num_pipeline_stages=int(num_pipeline_stages),
                    num_parallelized_experts=int(num_parallelized_experts),
                )
            else:
                new_weights = unpack_moe_weights_from_expert_parallel(weights)

            if new_weights is weights:
                continue
            _replace_moe_weight_aliases(model, weights, new_weights)
            del weights
            gc.collect()
            transformed = True
        return model, transformed


class OptimizedMoETransform(PytorchTransform):
    """Compatibility facade for MoE mapping discovery, weights, and export config."""

    @classmethod
    def apply(
        cls,
        model: nn.Module,
        *,
        prefill_only: bool = False,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = DEFAULT_AIC_NUM_CORES,
        qaic_config: dict | None = None,
        prefill_seq_len: int | None = None,
        hash_params: dict | None = None,
    ) -> tuple[nn.Module, bool]:
        model, mapped = OptimizedMoEMapperTransform.apply(model)
        model, external_mapped = ExternalOptimizedMoEMapperTransform.apply(model)
        if not (mapped or external_mapped):
            return model, False

        model, weights_ready = OptimizedMoEWeightsTransform.apply(model)
        model, export_configured = OptimizedMoEExportConfigTransform.apply(
            model,
            prefill_only=prefill_only,
            batch_size=batch_size,
            num_devices=num_devices,
            num_cores=num_cores,
            qaic_config=qaic_config,
            prefill_seq_len=prefill_seq_len,
            hash_params=hash_params,
        )
        _, expert_parallel_weights_ready = OptimizedMoEExpertParallelWeightsTransform.apply(model)
        return model, mapped or external_mapped or weights_ready or export_configured or expert_parallel_weights_ready


class SimpleDecodeMoeTransform(OptimizedMoETransform):
    """Constructor-time MoE transform that uses legacy decode defaults."""

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        return OptimizedMoETransform.apply(
            model,
            prefill_only=False,
        )
