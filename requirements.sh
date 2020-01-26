virtualenv -p python3 env
source env/bin/activate
pip3 install jupyter ipython numpy matplotlib pandas tensorflow nltk
pip install -U spacy
python -m spacy download en_core_web_sm
pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html