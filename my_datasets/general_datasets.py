import torch
from PIL import Image
from torch.utils.data import Dataset

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


class CropDataset(torch.utils.data.Dataset):
        def __init__(self, crops):
            self.crops = crops

        def __len__(self):
            return len(self.crops)

        def __getitem__(self, idx):
            return Image.fromarray(self.crops[idx])