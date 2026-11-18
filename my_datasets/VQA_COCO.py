import os
import json
from PIL import Image
from torch.utils.data import Dataset
from IPython.display import display


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