git config --global user.name "HP2706"
git config --global user.email "hp2706@gmail.com"

conda init
source ~/.bashrc
pip install uv
uv pip install -e .

export WANDB_API_KEY='a3469eb2df23f67e4d6907ebacf50ffb4ee664f7' 