# Text-to-Speech Classes


## Pipeline API

(QEffChatterboxT3)=
### `QEffChatterboxT3`

```{eval-rst}
.. autoclass:: QEfficient.tts.pipelines.pipeline_t3.QEffChatterboxT3
   :members:
   :no-show-inheritance:
```

---

(QEffS3GenFlowEncoder)=
### `QEffS3GenFlowEncoder`, `QEffS3GenFlowEstimator`, `QEffHiFTF0Predictor`, `QEffHiFTDecoder`

```{eval-rst}
.. autoclass:: QEfficient.tts.pipelines.pipeline_module.QEffS3GenFlowEncoder
   :no-show-inheritance:
.. autoclass:: QEfficient.tts.pipelines.pipeline_module.QEffS3GenFlowEstimator
   :no-show-inheritance:
.. autoclass:: QEfficient.tts.pipelines.pipeline_module.QEffHiFTF0Predictor
   :no-show-inheritance:
.. autoclass:: QEfficient.tts.pipelines.pipeline_module.QEffHiFTDecoder
   :no-show-inheritance:
```

---


## Model Classes

(QEffChatterboxMultilingualTTS)=
### `QEffChatterboxMultilingualTTS`

Resemble AI Chatterbox Multilingual (23 languages) on Cloud AI 100. Requires Chatterbox at git commit `5de7a54`,
installed with `--no-deps` (see `examples/audio/README.md`).

```python
from QEfficient import QEffChatterboxMultilingualTTS

pipeline = QEffChatterboxMultilingualTTS.from_pretrained("ResembleAI/chatterbox", t3_model="v3")
pipeline.compile()
output = pipeline.generate("Bonjour, comment allez-vous?", language_id="fr", seed=0)
print(output)  # per-module times; output.audio is the waveform at output.sample_rate
```

See examples:

- `examples/audio/text_to_speech.py`

```{eval-rst}
.. autoclass:: QEfficient.tts.pipelines.chatterbox.pipeline_chatterbox.QEffChatterboxMultilingualTTS
   :members: from_pretrained, export, compile, generate
   :no-show-inheritance:
```

(QEffTTSOutput)=
### `QEffTTSOutput`

```{eval-rst}
.. autoclass:: QEfficient.tts.pipelines.pipeline_utils.QEffTTSOutput
   :members:
   :no-show-inheritance:
```
