git config --global user.name "HP2706"
git config --global user.email "hp2706@gmail.com"

conda init
source ~/.bashrc
pip install uv
uv pip install -e .

# Add environment variables to ~/.bashrc
echo '
# Environment variables
export WANDB_API_KEY="a3469eb2df23f67e4d6907ebacf50ffb4ee664f7"
export HUGGINGFACE_API_KEY="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"
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER=1
' >> ~/.bashrc

# Source the updated .bashrc
source ~/.bashrc

# Create a jupyter kernel that includes these environment variables
python -m ipykernel install --user --name=nlp_course_env

# Create a jupyter config file if it doesn't exist
jupyter notebook --generate-config

# Add code to load environment variables in Jupyter
echo "
import os
os.environ['WANDB_API_KEY'] = 'a3469eb2df23f67e4d6907ebacf50ffb4ee664f7'
os.environ['HUGGINGFACE_API_KEY'] = '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'
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
" >> ~/.ipython/profile_default/startup/load_env.py

# Restart the Jupyter server
jupyter notebook stop
jupyter notebook