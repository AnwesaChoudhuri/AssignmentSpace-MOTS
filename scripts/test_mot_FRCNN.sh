
#srun -p terramepp --gres=gpu:1 --mem=30g --pty /bin/bash
#module load PyTorch/1.4.0-IGB-gcc-4.9.4-Python-3.6.1
#module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1 
#module load cocoapi


cd ../../

for vid in 0 1 2 3 4 5 6
do
	echo ${vid}
	python Structured_MOTS/structured_mots_main.py config_test_mot_FRCNN ${vid}

done

#########

cd Structured_MOTS/external/MOTChallengeEvalKit/MOT/
seqName="MOT17-02-FRCNN"

image_dir="../../../../data/MOT17/train/"${seqName}"/img1"
output_dir="../../../../output/MOT17/train/tracking/FRCNN_withoptflow/losses_tracklambda/test/epoch5/vid"
FilePath="../../../../output/MOT17/train/tracking/FRCNN_withoptflow/losses_tracklambda/test/epoch5/Instances_txt/"${seqName}".txt"

python MOTVisualization.py --image_dir ${image_dir} --output_dir ${output_dir} --FilePath ${FilePath} --mode "None" --seqName ${seqName}

########

cd ../../../../Structured_MOTS/external/TrackEval/
python scripts/run_mot_challenge.py --USE_PARALLEL False --TRACKERS_TO_EVAL epoch5
