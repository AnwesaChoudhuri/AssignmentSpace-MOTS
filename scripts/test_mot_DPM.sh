
#srun -p terramepp --gres=gpu:1 --mem=30g --pty /bin/bash
#module load PyTorch/1.4.0-IGB-gcc-4.9.4-Python-3.6.1
#module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1 
#module load cocoapi


cd ../../

for vid in 0 1 2 3 4 5 6
do
	echo ${vid}
	python Structured_MOTS/structured_mots_main.py config_test_mot_DPM ${vid}

done

#cd Structured_MOTS/external/mots_tools
#datadir="../../../data/KITTI_MOTS/val/"
#opdir="../../../output/KITTI_MOTS/val/tracking/pointtrack/losses_seg_tracktheta_tracklambda/test/epoch5/"
#python mots_eval/eval.py ${opdir}Instances_txt/ ${datadir}instances_txt/ mots_eval/val.seqmap
#python mots_vis/visualize_mots.py ${opdir}Instances_txt/ ${datadir}images/ ${opdir}Superimposed/ mots_eval/val.seqmap

