import timm
import torch
import torch.nn as nn
from utils.lxt.explicit.models.vilt import ViltForQuestionAnswering, attnlrp as lrp_rule_composite
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModelForImageClassification, Qwen2VLForConditionalGeneration


def create_resnet50(device="cpu"):
    weights = ResNet50_Weights.IMAGENET1K_V2

    model = resnet50(weights=weights)

    model.name = "resnet_50"
    model.classes = weights.meta["categories"].copy()

    return model.eval().to(device)


def create_vgg16(device="cpu"):
    model = AutoModelForImageClassification.from_pretrained("timm/vgg16.tv_in1k")

    model.name = "vgg16"

    id2label = model.config.id2label
    model.classes = [id2label[i] for i in range(len(id2label))]

    return model.to(device).eval()


def create_vit(device="cpu"):
    model = AutoModelForImageClassification.from_pretrained("timm/vit_small_patch16_224.augreg_in21k_ft_in1k")
    timm_version = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    model.name = "vit"
    model.patch_embed = timm_version.patch_embed

    id2label = model.config.id2label
    model.classes = [id2label[i] for i in range(len(id2label))]

    return model.to(device).eval()


def create_vilt(device="cpu"):
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    )

    model.name = "vilt"

    id2label = model.config.id2label
    model.classes = [id2label[i] for i in range(len(id2label))]

    return model.to(device).eval()


def create_qwen(device="cpu"):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        # device_map="auto",  # the accelerate library automatically decides where to place the model, i.e.  GPU or CPU
        torch_dtype=torch.float16,
    )

    model.name = "qwen"
    model.classes = None

    return model.to(device).eval()


class ViltCRPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def hf_model(self):
        return self.model

    def forward(
            self,
            pixel_values,
            inputs_embeds,
            attention_mask=None,
            token_type_ids=None,
            pixel_mask=None,
    ):
        return self.model(
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_mask=pixel_mask,
            return_dict=True,
        ).logits