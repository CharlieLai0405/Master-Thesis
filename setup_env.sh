#!/bin/bash

# 1. 建立並啟用 Conda 環境
echo "正在建立 Conda 環境: anomaly_gpt..."
conda create -n anomaly_gpt python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate anomaly_gpt

# 2. 安裝核心 Torch 與套件 (照你的清單)
echo "正在安裝核心套件..."
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install deepspeed==0.9.2

# 3. 批次安裝其餘套件
echo "正在安裝剩餘套件..."
pip install accelerate==1.10.1 aiofiles==23.2.1 altair==5.5.0 annotated-doc==0.0.4 \
annotated-types==0.7.0 anyio==4.12.1 asttokens==3.0.1 attrs==26.1.0 av==15.1.0 \
charset-normalizer==3.4.6 click==8.1.8 contourpy==1.3.0 cycler==0.12.1 \
decorator==5.2.1 easydict==1.10 einops==0.6.1 exceptiongroup==1.3.1 \
executing==2.2.1 fastapi==0.125.0 ffmpy==1.0.0 filelock==3.19.1 \
fonttools==4.60.2 fsspec==2025.10.0 ftfy==6.1.1 fvcore==0.1.5.post20221221 \
gradio==3.41.2 gradio-client==0.5.0 h11==0.16.0 h5py==3.9.0 hf-xet==1.4.2 \
hjson==3.1.0 httpcore==1.0.9 httpx==0.28.1 huggingface-hub==0.36.2 \
imageio==2.37.2 importlib-metadata==8.7.1 importlib-resources==6.5.2 \
iopath==0.1.10 ipdb==0.13.13 ipython==8.18.1 jedi==0.19.2 jinja2==3.1.6 \
joblib==1.5.3 jsonschema==4.25.1 jsonschema-specifications==2025.9.1 \
kiwisolver==1.4.7 kornia==0.7.0 latex2mathml==3.78.1 lazy-loader==0.5 \
markdown==3.9 markdown-it-py==3.0.0 markupsafe==2.1.5 matplotlib==3.7.2 \
matplotlib-inline==0.2.1 mdtex2html==1.2.0 mdurl==0.1.2 mpmath==1.3.0 \
msgpack==1.1.2 narwhals==2.18.0 networkx==3.2.1 ninja==1.13.0 \
nvidia-cusparselt-cu12==0.7.1 open3d-python==0.3.0.0 opencv-python==4.8.0.74 \
orjson==3.11.5 packaging==26.0 pandas==2.3.3 parameterized==0.9.0 \
parso==0.8.6 peft==0.3.0 pexpect==4.9.0 portalocker==3.2.0 \
prompt-toolkit==3.0.52 psutil==7.2.2 ptyprocess==0.7.0 pure-eval==0.2.3 \
py-cpuinfo==9.0.0 pydantic==1.10.26 pydantic-core==2.41.5 pydub==0.25.1 \
pygments==2.19.2 pyparsing==3.0.9 python-dateutil==2.9.0.post0 \
python-multipart==0.0.20 pytorchvideo==0.1.5 pytz==2026.1.post1 \
pyyaml==6.0.1 referencing==0.36.2 regex==2022.10.31 rich==14.3.3 \
rpds-py==0.27.1 safetensors==0.7.0 scikit-image==0.24.0 scikit-learn==1.3.0 \
scipy==1.13.1 semantic-version==2.10.0 sentencepiece==0.2.1 \
shellingham==1.5.4 six==1.17.0 stack-data==0.6.3 starlette==0.49.3 \
sympy==1.14.0 tabulate==0.9.0 termcolor==2.3.0 threadpoolctl==3.6.0 \
tifffile==2024.8.30 timm==0.6.7 tokenizers==0.13.3 tomli==2.4.0 \
tqdm==4.64.1 traitlets==5.14.3 transformers==4.29.1 triton==3.4.0 \
typer==0.23.2 typing-inspection==0.4.2 tzdata==2025.3 uvicorn==0.39.0 \
wcwidth==0.6.0 websockets==11.0.3 yacs==0.1.8 zipp==3.23.0

# 4. 關鍵修正：修復 pytorchvideo 與 torchvision 的相容性
echo "正在修復 pytorchvideo 原始碼 Bug..."
VENV_PATH=$(python -c "import pytorchvideo; import os; print(os.path.dirname(pytorchvideo.__file__))")
grep -rrl "torchvision.transforms.functional_tensor" $VENV_PATH | xargs sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g'

echo "環境安裝完成！請使用 'conda activate anomaly_gpt' 啟用環境。"
