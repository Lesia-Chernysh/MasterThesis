import kagglehub
from numpy.random import _generator
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImagenetMini(Dataset):
    def __init__(self, wordnet_ids_to_classes_path, transform=None):
        """
        Since ImageNetMini has to labels, those can be retrieved in 2 ways:
            1) through the path to the image, since it contains word net index
            2) they can be predicted by a model
        For conditioning attribution on a certain class, it is not useful to know the true label of the image, since
        the classes of the classifiers are probably different.
        :param transform: depends on the model
        :param wordnet_ids_to_classes_path:
        :param predicted_labes: optional, to make attribution easier.
        :return: path, label
        """
        self.root = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
        self.dataset = ImageFolder(
            root=self.root ,
            transform=transform,
        )
        if wordnet_ids_to_classes_path is not None:
            self.idx2labels = self._get_class_names(wordnet_ids_to_classes_path)
            self.class_n = len(self.idx2labels)
        else:
            self.idx2labels = None
            self.class_n = None
            print("WARNING: imagenet-mini doesn't contain labels. Labels are set to None. Number of clusses is set to None.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        if self.idx2labels is not None:
            # self.dataset.samples[idx]] is a list of length = idx. idx is defined when I slice over the ds
            path, _ = self.dataset.samples[idx]
            try:
                label = self.idx2labels[path.split("\\")[-2]]
            except IndexError:
                label = self.idx2labels[path.split("/")[-2]]
            return image, label
        else:
            return image, None

    def _get_class_names(self, wordnet_ids_to_classes_path) -> dict:
        """
        Returns a mapping between wordnet_ids and class names.
        :param wordnet_ids_to_classes_path:
        :return: dict[str, str]
        """
        if not wordnet_ids_to_classes_path.endswith(".txt"):
            print("""
            Wordent ids -> classes mapping is implemented only for .txt.
            Expected structure is:
                wordent id: class name
            Example:
                n02666196: abacus
            """)
            raise NotImplementedError
        class_names = {}
        with open(wordnet_ids_to_classes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                wordn_idx, class_name = line.split(":", 1)  # Split only on the first colon
                class_names[wordn_idx.strip()] = class_name.strip()
        return class_names


class ImagenetMiniPredLabels(ImagenetMini):
    def __init__(self, wordnet_ids_to_classes_path, predicted_labels):
        super().__init__(wordnet_ids_to_classes_path)
        self.predicted_labels = predicted_labels

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.predicted_labels[idx]


class ImagenetMiniVQA(Dataset):
    def __init__(self, dataset, predicted_labels, question="What is depicted on the image?"):
        self.dataset = dataset
        self.question = question
        self.classes = None
        self.predicted_labels = predicted_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        label = self.predicted_labels[idx]

        return image, self.question, label