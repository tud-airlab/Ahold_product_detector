import time
from pathlib import Path
from typing import Optional

import rospy
import torch

import torch.nn.functional as F
import torchvision
from torch import nn

import pmf.models.vision_transformer as vit
from pmf.models import ProtoNet
from pmf_data_helpers import IMAGE_LOADER, ImageLoader


class StaticProtoNet(ProtoNet):
    def __init__(self, model: dict):
        backbone = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        super().__init__(backbone)
        super().load_state_dict(model)

        self.prototypes = None

    def update_prototypes(self, prototypes):
        self.prototypes = F.normalize(prototypes, p=2, dim=prototypes.dim() - 1, eps=1e-12)

    def cos_classifier(self, features, **kwargs):
        if self.prototypes is None:
            raise Exception("No prototype set")
        else:
            f = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
            cls_scores = f @ self.prototypes.transpose(0, 1)
            cls_scores = self.scale_cls * (cls_scores + self.bias)
            cls_scores = self.sigmoid(cls_scores)
            return cls_scores


class ProtoTypeLoader:
    def __init__(self, feature_extractor: torch.nn, image_loader: ImageLoader,
                 prototype_dict: Optional[dict] = None,
                 path_to_dataset: Optional[Path] = None, device: str = "cuda:0"):
        self.feature_extractor = feature_extractor
        self.image_loader = image_loader
        self.device = device
        self.prototype_dict = prototype_dict
        if self.prototype_dict is None:
            self.prototype_dict = self.fill_prototype_dict(batch_size=150, path_to_dataset=path_to_dataset)

    def load_prototypes(self, class_to_find: str, amount_of_prototypes: int = 5):
        prototype_dict = self.prototype_dict.copy()
        class_to_find_tensor = prototype_dict.pop(class_to_find)
        class_to_find_tensor = class_to_find_tensor.view(-1, class_to_find_tensor.shape[0])
        other_prototypes_tensor = torch.empty(size=(len(prototype_dict), class_to_find_tensor.shape[1]),
                                              dtype=torch.float, device=self.device, requires_grad=False)
        for i, key in enumerate(prototype_dict.keys()):
            other_prototypes_tensor[i] = prototype_dict[key]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        class_similarity = cos(class_to_find_tensor, other_prototypes_tensor)
        _, top_indices = torch.topk(class_similarity, amount_of_prototypes - 1)

        class_prototype_tensor = torch.cat((class_to_find_tensor, other_prototypes_tensor[top_indices]), dim=0)
        classes = [class_to_find] + [list(prototype_dict.keys())[idx] for idx in top_indices.tolist()]
        return classes, class_prototype_tensor

    def fill_prototype_dict(self, path_to_dataset, batch_size=150):
        prototype_dict = {}
        if not path_to_dataset.exists():
            raise Exception("Please specify a valid path to dataset")
        for class_path in path_to_dataset.iterdir():
            print("Processing:", class_path)
            if class_path.is_dir():
                images = [image for image in class_path.iterdir() if image.is_file()]
                images_feature_tensor = torch.empty(size=(len(images), 384),
                                                    dtype=torch.float, device=self.device, requires_grad=False)
                single_image_tensor = torch.empty(size=(3, self.image_loader.image_size, self.image_loader.image_size),
                                                  dtype=torch.float, device=self.device,
                                                  requires_grad=False)

                images_in = 0
                while images:
                    image_batch, images = images[:batch_size], images[batch_size:]
                    batch_image_tensor = torch.empty(
                        size=(len(image_batch), 3, self.image_loader.image_size, self.image_loader.image_size),
                        dtype=torch.float, device=self.device, requires_grad=False)
                    for i, image in enumerate(image_batch):
                        batch_image_tensor[i] = single_image_tensor.copy_(self.image_loader(image))

                    _, C, H, W = batch_image_tensor.shape
                    with torch.no_grad():
                        image_features = self.feature_extractor.forward(batch_image_tensor.view(-1, C, H, W))
                    images_feature_tensor[images_in:images_in + len(image_batch)] = image_features
                    images_in += len(image_batch)
                class_prototype = torch.mean(images_feature_tensor, dim=0)
                prototype_dict[class_path.name] = class_prototype
        return prototype_dict


class PMF:
    def __init__(self, pmf_model_path: Path, image_loader: ImageLoader, classification_confidence_threshold: float,
                 path_to_dataset: Optional[Path] = None, reload_prototypes: bool = False, device="cuda:0"):
        self.pmf_path = pmf_model_path
        pmf_dict = torch.load(self.pmf_path)
        if 'image_loader' not in pmf_dict or image_loader != pmf_dict['image_loader']:
            reload_prototypes = True

        self.image_loader = image_loader
        self.device = device

        if "model" not in pmf_dict:
            raise Exception("Please provide path to a model")
        self.protonet = StaticProtoNet(model=pmf_dict["model"])
        self.protonet.to(self.device)

        if "prototype_dict" not in pmf_dict or reload_prototypes is True:
            self.prototype_loader = ProtoTypeLoader(feature_extractor=self.protonet.backbone,
                                                    path_to_dataset=path_to_dataset,
                                                    image_loader=self.image_loader,
                                                    device=self.device)
            self.save_model_dict(self.pmf_path)
        else:
            self.prototype_loader = ProtoTypeLoader(feature_extractor=self.protonet.backbone,
                                                    prototype_dict=pmf_dict["prototype_dict"],
                                                    image_loader=image_loader,
                                                    device=self.device)

        self.class_list = None

        if 0 <= classification_confidence_threshold < 1:
            self.clf_confidence_threshold = classification_confidence_threshold
        else:
            raise Exception("No valid confidence threshold supplied")

    def save_model_dict(self, path: Path):
        print("Saving prototype loader")
        dict_to_save = {"model": self.protonet.state_dict(),
                        "prototype_dict": self.prototype_loader.prototype_dict,
                        "image_loader": self.image_loader}
        torch.save(dict_to_save, path)

    def __call__(self, image_tensors: torch.Tensor, debug: bool = False):
        with torch.no_grad():
            image_features = self.protonet.backbone.forward(image_tensors)
            predictions = self.protonet.cos_classifier(image_features)
        scores, indices = torch.max(predictions, dim=1)
        scores[scores < self.clf_confidence_threshold] = 0
        if debug:
            classes = [self.class_list[idx] if score >= self.clf_confidence_threshold else "No Class" for idx, score in
                       zip(indices, scores)]
            return scores, classes
        else:
            scores[indices != 0] = 0
            classes = [self.get_class_to_find() if score != 0 else "_" for score in scores.tolist()]
            return scores, classes

    def set_class_to_find(self, class_to_find):
        current_class = self.class_list[0] if self.class_list is not None else None
        if current_class != class_to_find:
            self.class_list, class_prototypes = self.prototype_loader.load_prototypes(class_to_find)
            self.protonet.update_prototypes(class_prototypes)
            rospy.loginfo(f"Detection class set from {current_class} to {class_to_find}")

    def get_class_to_find(self):
        return self.class_list[0] if self.class_list is not None else None
