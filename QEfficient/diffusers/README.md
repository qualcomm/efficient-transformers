
<div align="center">


# **Diffusion Models on Qualcomm Cloud AI 100**


<div align="center">

### 🎨 **Experience the Future of AI Image Generation**

* Optimized for Qualcomm Cloud AI 100*

<img src="../../docs/image/girl_laughing.png" alt="Sample Output" width="400">

**Generated with**: `black-forest-labs/FLUX.1-schnell` • `"A girl laughing"` • 4 steps • 0.0 guidance scale •  ⚡



</div>



[![Diffusers](https://img.shields.io/badge/Diffusers-0.35.1-orange.svg)](https://github.com/huggingface/diffusers)
</div>

---

## ✨ Overview

QEfficient Diffusers brings the power of state-of-the-art diffusion models to Qualcomm Cloud AI 100 hardware for text-to-image generation. Built on top of the popular HuggingFace Diffusers library, our optimized pipeline provides seamless inference on Qualcomm Cloud AI 100 hardware.

## 🛠️ Installation

### Prerequisites

Ensure you have Python 3.10+ and the required dependencies:

```bash
# Create Python virtual environment (Recommended Python 3.10)
sudo apt install python3.10-venv
python3.10 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip
```

### Install QEfficient

Install QEfficient from the official `quic/efficient-transformers` repository or a wheel built from that source.
Avoid `pip install qefficient` until maintainers confirm an official PyPI release under their control.
For reproducible installs, pin the Git URL to a reviewed commit using `@<commit-sha>`.

```bash
# Install from GitHub (includes diffusers support)
pip install git+https://github.com/quic/efficient-transformers

# Or build from source
git clone https://github.com/quic/efficient-transformers.git
cd efficient-transformers
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/qefficient-0.0.1.dev0-py3-none-any.whl
```

---

## 🎯 Supported Models
- ✅ [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- ✅ [`lightx2v/Wan2.2-Lightning`](https://huggingface.co/lightx2v/Wan2.2-Lightning)

---


## 📚 Examples

Check out our comprehensive examples in the [`examples/diffusers/`](../../examples/diffusers/) directory:

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.



---

## 🙏 Acknowledgments

- **HuggingFace Diffusers**: For the excellent foundation library
---

## 📞 Support

- 📖 **Documentation**: [https://quic.github.io/efficient-transformers/](https://quic.github.io/efficient-transformers/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/quic/efficient-transformers/issues)

---
