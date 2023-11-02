pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install --editable ./
python setup.py build_ext --inplace 
pip install nltk
pip install boto3
pip install six
pip install requests
pip install sacremoses
pip install scikit-learn
pip install packaging
pip install filelock
pip install huggingface_hub
pip install tokenizers
pip install spacy
pip install pyrouge-master
