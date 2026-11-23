# Example Input Notices

The fixed inputs are included for single-example inference, not as an
evaluation suite or a representative throughput benchmark.

## Triangle Image and Question

The default image question and linked image come from MathLLMs/MathVision,
revision `2837ddb3f13abaf6b3997c12d80753e5470bd46a`, testmini item `8`.
The question's `<image1>` placeholder is omitted because the processor inserts
the image tokens. A final-answer formatting instruction is appended. The image
is not redistributed in this repository; the example downloads `images/8.jpg`
from the upstream repository at commit
`a1f5cc3add200c0cd080fad463e500f44ef1fb41`.

Source: https://huggingface.co/datasets/MathLLMs/MathVision
Upstream license: https://github.com/mathllm/MATH-V/blob/main/LICENSE
The dataset card identifies the license as MIT. The upstream license notice is
reproduced below as published.

MIT License

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Coding Prompt

`DEFAULT_TEXT_PROMPT` in `utils.py` includes the function specification from
OpenAI HumanEval task 0, with a surrounding instruction to complete the code.

Source: https://github.com/openai/human-eval
Upstream license: https://github.com/openai/human-eval/blob/master/LICENSE

The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
