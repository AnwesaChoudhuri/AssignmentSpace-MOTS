
#srun -p terramepp --gres=gpu:1 --mem=30g --pty /bin/bash
#module load PyTorch/1.4.0-IGB-gcc-4.9.4-Python-3.6.1
#module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1
#module load cocoapi


cd ../../

for vid in 5 7 # 11 12  14 15 #1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 #
do
	echo ${vid}
	python Structured_MOTS/structured_mots_main_online.py config_test_kittimots_online ${vid}

done

cd Structured_MOTS_online/external/mots_tools
datadir="../../../data/KITTI_MOTS/val/"
opdir="../../../output/KITTI_MOTS/val/tracking/pointtrack_online/losses_tracklambda/test/epoch25/"
python mots_eval/eval.py ${opdir}Instances_txt/ ${datadir}instances_txt/ mots_eval/val.seqmap
#python mots_vis/visualize_mots.py ${opdir}Instances_txt/ ${datadir}images/ ${opdir}Superimposed/ mots_eval/val.seqmap
