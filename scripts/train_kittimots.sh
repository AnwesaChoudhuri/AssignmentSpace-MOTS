
srun -p terramepp --gres=gpu:1 --mem=30g --pty /bin/bash
module load PyTorch/1.4.0-IGB-gcc-4.9.4-Python-3.6.1
module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1 
module load cocoapi


cd ../../
python Structured_MOTS/structured_mots_main.py config_train_kittimots_jt #config_train_kittimots

