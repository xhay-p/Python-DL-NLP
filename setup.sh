virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm