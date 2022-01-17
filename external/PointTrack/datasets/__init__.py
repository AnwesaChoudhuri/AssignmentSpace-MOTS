from .KittiMOTSDataset import *


def get_dataset(name, dataset_opts):
    if name == "mots_test":
        return MOTSTest(**dataset_opts)
        
    elif name == "mots_cars_val":
        return MOTSCarsVal(**dataset_opts)
    elif name == "mots_persons_val":
        return MOTSPersonsVal(**dataset_opts)

    elif name == "mots_track_val_env_offset":
        return MOTSTrackCarsValOffset(**dataset_opts)
    elif name == "mots_track_val_env_offset_person":
        return MOTSTrackPersonValOffset(**dataset_opts)

    elif name == "mots_track_cars_train":
        return MOTSTrackCarsTrain(**dataset_opts)
    elif name == "mots_track_person_train":
        return MOTSTrackPersonTrain(**dataset_opts)

    elif name == "mots_cars":
        return MOTSCars(**dataset_opts)
    elif name == "mots_persons":
        return MOTSPersons(**dataset_opts)

    else:
        raise RuntimeError("Dataset {} not available".format(name))
