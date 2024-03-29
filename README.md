# Assignment-Space-Based Multi-Object Tracking and Segmentation (ICCV 2021)

[Anwesa Choudhuri](https://www.linkedin.com/in/anwesachoudhuri/), [Girish Chowdhary](https://ece.illinois.edu/about/directory/faculty/girishc), [Alexander G. Schwing](https://alexander-schwing.de/)

[[`Publication`](https://openaccess.thecvf.com/content/ICCV2021/html/Choudhuri_Assignment-Space-Based_Multi-Object_Tracking_and_Segmentation_ICCV_2021_paper.html)] [[`Project`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/)] [[`BibTeX`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/bib.txt)]

<div align="center">
  <img src="./imgs/car.png" width="40%" height="40%"/>
  <img src="./imgs/person.png" width="40%" height="40%"/>
  <img src="./imgs/3d_1.png" width="40%" height="40%"/>
  <img src="./imgs/3d_2.png" width="40%" height="40%"/>
</div><br/>

## Getting Started

### Prerequisites:
1. Virtual environment with Python 3.6
2. Pytorch 1.3.1 
3. Other requirements:
```
$ pip install -r requirements.txt
```

### Dataset:
[KITTI Images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) + [Annotations](https://www.vision.rwth-aachen.de/media/resource_files/instances.zip)

Structure should be the following:
```
AssignmentSpace-MOTS
│   data
│   │   KITTI_MOTS
│   │   │    train
│   │   │   │   images
│   │   │   │    instances_txt
│   │   │    val
│   │   │   │    images
│   │   │   │    instances_txt
│   │   │    test
│   │   │   │    images
```

Note: Please use [RAFT](https://github.com/princeton-vl/RAFT) to run optical flow between consecutive and alternate files and save them as numpy files. Using optical flow is optional. 

Structure should be the following:

```
AssignmentSpace-MOTS
│   data
│   │   KITTI_MOTS
│   │   │    {train,val,test}
│   │   │   │   RAFT_optical_flow
│   │   │   │   │   flow_skip0
│   │   │   │   │   flow_skip1
```


## Saved detections and models

Detections and saved models are stored [here](https://drive.google.com/drive/folders/1C8E__yOR2D36oj-fq8cYRuteipcsXyTB?usp=share_link). Download them in the homedir.

Structure should be the following:

```
AssignmentSpace-MOTS
│   saved_models
│   detections

```

## Testing

To test for cars:
./scripts/test_kittimots.sh

To test for pedestrians:
./scripts/test_kittimots_ped.sh


## Citing Assignment-Space-Based MOTS

If you find the code or paper useful, please cite the following BibTeX entry.

```BibTeX
@InProceedings{Choudhuri_2021_ICCV,
    author    = {Choudhuri, Anwesa and Chowdhary, Girish and Schwing, Alexander G.},
    title     = {Assignment-Space-Based Multi-Object Tracking and Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13598-13607}
}
```

## Acknowledgement

This work is supported in party by Agriculture and Food Research Initiative (AFRI) grant no. 2020-67021-32799/project accession no.1024178 from the USDA National Institute of Food and Agriculture: NSF/USDA National AI Institute: AIFARMS. We also thank the Illinois Center for Digital Agriculture for seed funding for this project. 

