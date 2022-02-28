# Assignment-Space-Based Multi-Object Tracking and Segmentation

[Anwesa Choudhuri](https://www.linkedin.com/in/anwesachoudhuri/), [Girish Chowdhary](https://ece.illinois.edu/about/directory/faculty/girishc), [Alexander G. Schwing](https://alexander-schwing.de/)

[[`Publication`](https://openaccess.thecvf.com/content/ICCV2021/html/Choudhuri_Assignment-Space-Based_Multi-Object_Tracking_and_Segmentation_ICCV_2021_paper.html)] [[`Project`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/)] [[`BibTeX`](https://anwesachoudhuri.github.io/Assignment-Space-based-MOTS/bib.txt)]


## Getting Started

### Prerequisites:
Pytorch 1.3.1, please set up an virtual env and run:
```
$ pip install -r requirements.txt
```

### Dataset:
[KITTI Images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) + [Annotations](https://www.vision.rwth-aachen.de/media/resource_files/instances.zip)

Structure should be the following:

data
    KITTI_MOTS
        train
            images
            instances_txt
        val
            images
            instances_txt
        test
            images

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
