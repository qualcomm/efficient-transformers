# Pre-requisites
System Requirements:
1. [Supported Linux OS](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/#operating-systems) - Ubuntu, RHEL and AWS Linux
2. [Cloud AIxxx (AI100, AI200 and so on) Platform SDK installed](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/#platform-sdk) 
3. [SDK Pre-requisites](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Pre-requisites/pre-requisites/) 
4. [Multi-device support enabled for model sharding](https://github.com/quic/cloud-ai-sdk/tree/1.12/utils/multi-device)

---

# Installation 

## 1. Download Apps SDK
   * [Cloud AIxxx (AI100, AI200 and so on) Apps SDK install](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/)  

Uninstall existing Apps SDK
```
sudo ./uninstall.sh
```
Run the install.sh script as root or with sudo to install with root permissions.
```
sudo ./install.sh --enable-qeff
source  /opt/qti-aic/dev/python/qeff/bin/activate
```
On successful installation, the contents are stored to the /opt/qti-aic path under the dev and exec directories:
```
dev exec integrations scripts
```
Check the Apps SDK version with the following command
```
sudo /opt/qti-aic/tools/qaic-version-util --apps
```
Apply chmod commands
```
sudo chmod a+x /opt/qti-aic/dev/hexagon_tools/bin/*
sudo chmod a+x /opt/qti-aic/exec/*
```

---

## 2. Install Efficient-Transformers

### Using GitHub Repository

Install QEfficient from the official `quic/efficient-transformers` repository or a wheel built from that source.
Avoid `pip install qefficient` until maintainers confirm an official PyPI release under their control.
For reproducible installs, pin the Git URL to a reviewed commit using `@<commit-sha>`.

```{warning}
Efficient Transformers have been validated to work with the same compatible SDK. Upgrading this may result in certain models becoming incompatible.
```

```bash
# Create Python virtual env and activate it. (Required Python 3.10)

python3.12 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient Repo.
pip install git+https://github.com/quic/efficient-transformers

``` 

---

# Sanity Check

After above installation methods, you can check if ``QEfficient`` is installed correctly by using
```bash
python -c "import QEfficient; print(QEfficient.__version__)"
```
If the above line executes successfully, you are good to go ahead and start deploying models on ``Cloud AIxxx (AI100, AI200 and so on)`` cards using ``QEfficient`` library.
