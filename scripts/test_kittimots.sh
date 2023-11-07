
for vid in 0 1 2 3 4 5 6 7 8
do
	echo ${vid}
	python structured_mots_main.py config_test_kittimots ${vid}

done

cd external/mots_tools
datadir="../../data/KITTI_MOTS/val/"
opdir="../../outputs/KITTI_MOTS/val/test/car/"
python mots_eval/eval.py ${opdir}Instances_txt/ ${datadir}instances_txt/ mots_eval/val.seqmap

## uncomment the next line if you want to visualize
# python mots_vis/visualize_mots.py ${opdir}Instances_txt/ ${datadir}images/ ${opdir}Superimposed/ mots_eval/val.seqmap
