virtualenv -p python3 env
source env/bin/activate
# For llama-cpp in Ubuntu
sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11
pip install -r requirements.txt
python -m spacy download en_core_web_sm