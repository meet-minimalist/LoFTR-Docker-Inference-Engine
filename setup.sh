##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-25 8:23:13 pm
# @copyright MIT License
#

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/kornia/kornia.git

cd kornia

cp ../patch.txt ./

git apply --ignore-space-change --ignore-whitespace patch.txt

pip install .

cd ..

rm -rf kornia

pip --no-cache-dir install -r requirements.txt

pip uninstall opencv-python -y

pip install opencv_python_headless==4.8.0.74

mkdir temp

cd temp 

wget https://raw.githubusercontent.com/kornia/data/main/matching/kn_church-2.jpg

wget https://raw.githubusercontent.com/kornia/data/main/matching/kn_church-8.jpg

cd ..

python export_model.py
