import torch
from PIL import Image
from torch.utils.data import Dataset
from interpretability.attribution_methods.semanticlens.utils.helper import restore_crop

class IndexedSampleDataset(Dataset):
    """
    Run only selected samples while retaining their original
    dataset indices.
    """

    def __init__(
        self,
        dataset,
        sample_ids,
    ):
        self.dataset = dataset
        self.sample_ids = list(sample_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample = self.dataset[sample_id]

        # My  datasets normally return (image, label).
        if (
            isinstance(sample, tuple)
            or isinstance(sample, list)
        ):
            image = sample[0]
            remaining_values = tuple(sample[1:])
        else:
            image = sample
            remaining_values = ()

        return (
            sample_id,
            image,
            *remaining_values,
        )



from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CropDataset(Dataset):
    def __init__(self, crop_db):
        self.crop_db = crop_db
        print("NEEW CROPDATASET")

    def __len__(self):
        return len(self.crop_db)

    def __getitem__(self, idx):
        el = self.crop_db[idx]

        crop = restore_crop(
            el["image_path"],
            el["bbox"]
        )

        # Convert NumPy array to PIL image
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(crop)

        return crop

