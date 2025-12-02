# 使用python310，cuda12.4
# 安装cuda12.1版本的torch与fbgemm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt