# Assignment-Space-Based Multi-Object Tracking and Segmentation (ICCV 2021)

[Anwesa Choudhuri](https://www.linkedin.com/in/anwesachoudhuri/), [Girish Chowdhary](https://ece.illinois.edu/about/directory/faculty/girishc), [Alexander G. Schwing](https://alexander-schwing.de/)

[[`Publication`](https://openaccess.thecvf.com/content/ICCV2021/html/Choudhuri_Assignment-Space-Based_Multi-Object_Tracking_and_Segmentation_ICCV_2021_paper.html)] [[`Project`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/)] [[`BibTeX`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/bib.txt)]

<div align="center">
  <img src="https://github.com/AnwesaChoudhuri/Assignment-Space-based-MOTS/blob/main/3d_2.png" width="50%" height="50%"/>
  <img src="https://github.com/AnwesaChoudhuri/Assignment-Space-based-MOTS/blob/main/3d_1.png" width="50%" height="50%"/>
  <img src="https://github.com/AnwesaChoudhuri/Assignment-Space-based-MOTS/blob/main/3d_3.png" width="50%" height="50%"/>
  <img src="https://github.com/AnwesaChoudhuri/Assignment-Space-based-MOTS/blob/main/3d_4.png" width="50%" height="50%"/>
</div><br/>



## Getting Started

### Prerequisites:
1. Virtual environment with Python3.6
2. Pytorch 1.3.1
3. Run the following:
```
$ pip install -r requirements.txt
```

### Dataset:
[KITTI Images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) + [Annotations](https://www.vision.rwth-aachen.de/media/resource_files/instances.zip)

Structure should be the following:
```
data
│   KITTI_MOTS
│   │    train
│   │   │   images
│   │   │    instances_txt
│   │    val
│   │   │    images
│   │   │    instances_txt
│   │    test
│   │   │    images
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
