import torch

from datasets.imagenet_mini import ImagenetMini, ImagenetMiniVQA, ImagenetMiniPredLabels
from interpretability.visualizations.visualization_functions import visualize_dataset_sample

# ------------- Test Datasets work ----------------------
imagenet_mini = ImagenetMini(wordnet_ids_to_classes_path="../datasets/input_files/wordnet_ids_to_classes.txt")
print("ImagenetMini, first sample. Original labels", imagenet_mini[0])

predictions_resnet_50 = "../datasets/input_files/resnet50_predicted_targets.pt"
predictions_resnet_50 = torch.load(predictions_resnet_50)
print("First 5 predicted labels", predictions_resnet_50[:5])

dataset = imagenet_mini.dataset
predicted = ImagenetMiniPredLabels(
    "../datasets/input_files/wordnet_ids_to_classes.txt",
    predictions_resnet_50)
print("ImagenetMini. Predicted labels", predicted[0])


vqa = ImagenetMiniVQA(
    dataset,
    predictions_resnet_50
)
print("ImagenetMini for VQA", vqa[0])


# --------------- Test visualization work --------------------
print("Visualizing first sample with a function")
visualize_dataset_sample(vqa[0])