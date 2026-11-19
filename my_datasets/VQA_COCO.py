import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class ImageFolderImitator():
    # to imitate ImageFolder object, so that VQACoco fits in existing pipeline

    # I need self.dataset.samples, which consists of (img_path, label) and has the length of image-question pairs.
    # The problem is that there are more questions than images, so the ids in sample ids correspond to the pair ids, not to the images ids alone

    def __init__(self, questions, imgs_path):
        self.questions = questions
        self.imgs_path = imgs_path
        self.samples = self.create_samples()

    def create_samples(self):
        samples = []

        for sample in self.questions:
            image_id = sample["image_id"]
            question = sample["question"]

            image_path = os.path.join(
                self.imgs_path,
                f"COCO_val2014_{image_id:012d}.jpg"
            )

            samples.append((image_path, 0))

        print(f"imitator. len samples: {len(samples)}")
        return samples


class VQACoco(Dataset):
    def __init__(self, root="/content/vqa", transform=None):
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, "val2014")

        questions_path = os.path.join(
            root, "v2_OpenEnded_mscoco_val2014_questions.json"
        )

        with open(questions_path, "r") as f:
            self.questions = json.load(f)["questions"]

        self.dataset = ImageFolderImitator(self.questions, self.image_dir)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        sample = self.questions[idx]

        image_id = sample["image_id"]
        question = sample["question"]

        image_path = os.path.join(
            self.image_dir,
            f"COCO_val2014_{image_id:012d}.jpg"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, question