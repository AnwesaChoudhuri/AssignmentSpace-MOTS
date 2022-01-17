
#srun -p terramepp --gres=gpu:1 --mem=30g --pty /bin/bash
#module load PyTorch/1.6.0-IGB-gcc-4.9.4-Python-3.6.1
#module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1 
#module load cocoapi


cd ../../

for vid in 0 1 2 3 4 5 6
do
	echo ${vid}
	python Structured_MOTS/structured_mots_main.py config_test_mot_2DMOT15train ${vid}

done

cd Structured_MOTS/external/MOTChallengeEvalKit/MOT
#python MOT/MOTVisualization.py --seqName "MOT17-02-FRCNN" --FilePath "../../../output/MOT17/train/tracking/FRCNN_withoptflow/losses_tracklambda/test/epoch5/Instances_txt/MOT17-02-FRCNN.txt" --image_dir "../../../../data/MOT17/train/MOT17-02-FRCNN/img1" --output_dir "../../../../output/MOT17/train/tracking/FRCNN_withoptflow/losses_tracklambda/test/epoch5/vid"

python MOTVisualization.py --seqName "ADL-Rundle-6" --FilePath "../../../../output/2DMOT2015/train/tracking/withoptflow/losses_tracklambda/test/epoch5/Instances_txt/ADL-Rundle-6.txt" --image_dir "../../../../data/MOTChallenge/2DMOT2015/train/ADL-Rundle-6/img1" --output_dir "../../../../output/2DMOT2015/train/tracking/withoptflow/losses_tracklambda/test/epoch5/vid"


cd ../../TrackEval
python -m pdb -c continue scripts/run_mot_challenge.py --USE_PARALLEL False --TRACKERS_TO_EVAL epoch5 --GT_FOLDER '../../../data/MOTChallenge/' --TRACKERS_FOLDER "../../../output/2DMOT2015/train/tracking/withoptflow/losses_tracklambda/test/" --BENCHMARK '2DMOT2015' --SPLIT_TO_EVAL 'train' --TRACKER_SUB_FOLDER 'Instances_txt'  --OUTPUT_SUB_FOLDER '' --SEQMAP_FOLDER "../MOTChallengeEvalKit/seqmaps/" --SEQMAP_FILE "../MOTChallengeEvalKit/seqmaps/2D_MOT_2015-train1.txt"

python -m pdb -c continue scripts/run_mot_challenge.py --USE_PARALLEL False --TRACKERS_TO_EVAL epoch5_interp --GT_FOLDER '../../../data/MOTChallenge/' --TRACKERS_FOLDER "../../../output/2DMOT2015/train/tracking/withoptflow/losses_tracklambda/test/" --BENCHMARK '2DMOT2015' --SPLIT_TO_EVAL 'train' --TRACKER_SUB_FOLDER 'Instances_txt'  --OUTPUT_SUB_FOLDER '' --SEQMAP_FOLDER "../MOTChallengeEvalKit/seqmaps/" --SEQMAP_FILE "../MOTChallengeEvalKit/seqmaps/2D_MOT_2015-train1.txt"


#python -m pdb -c continue scripts/run_mot_challenge.py --USE_PARALLEL False --TRACKERS_TO_EVAL epoch5 --GT_FOLDER '../../../data/MOTChallenge/' --TRACKERS_FOLDER "../../../output/2DMOT2015/other_trackers/" --BENCHMARK '2DMOT2015' --SPLIT_TO_EVAL 'train' --TRACKER_SUB_FOLDER 'Instances_txt'  --OUTPUT_SUB_FOLDER '' --SEQMAP_FOLDER "../MOTChallengeEvalKit/seqmaps/" --SEQMAP_FILE "../MOTChallengeEvalKit/seqmaps/2D_MOT_2015-train1.txt"



