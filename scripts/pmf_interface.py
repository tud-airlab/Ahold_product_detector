import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import rospy
import torch

import torch.nn.functional as F
from torch import nn

import pmf.models.vision_transformer as vit
from pmf.models import ProtoNet
from pmf_data_helpers import IMAGE_LOADER, ImageLoader
from roi_selector import ROISelector


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
        self.path_to_dataset = path_to_dataset
        self.roi_selector = ROISelector()
        self.unset_class = None
        if self.prototype_dict is None:
            self.prototype_dict = self._fill_prototype_dict(batch_size=150, path_to_dataset=path_to_dataset)

        self.prototypes = None
        self.classes = None

    def _load_prototypes_from_dict(self, class_name, amount_of_prototypes):
        prototype_dict = self.prototype_dict.copy()
        class_to_find_tensor = prototype_dict.pop(class_name)
        class_to_find_tensor = class_to_find_tensor.view(-1, class_to_find_tensor.shape[0])
        other_prototypes_tensor = torch.empty(size=(len(prototype_dict), class_to_find_tensor.shape[1]),
                                              dtype=torch.float, device=self.device, requires_grad=False)
        for i, key in enumerate(prototype_dict.keys()):
            other_prototypes_tensor[i] = prototype_dict[key]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        class_similarity = cos(class_to_find_tensor, other_prototypes_tensor)
        _, top_indices = torch.topk(class_similarity, amount_of_prototypes - 1)

        class_prototype_tensor = torch.cat((class_to_find_tensor, other_prototypes_tensor[top_indices]), dim=0)
        classes = [class_name] + [list(prototype_dict.keys())[idx] for idx in top_indices.tolist()]
        return classes, class_prototype_tensor

    def get_current_class(self):
        if self.classes is not None:
            return self.classes[0]

    def load_prototypes(self, barcode: str, amount_of_prototypes: int = 5):
        if self.get_current_class() is None or barcode not in self.get_current_class():
            for class_name in self.prototype_dict.keys():
                if barcode in class_name:
                    self.classes, prototypes = self._load_prototypes_from_dict(class_name, amount_of_prototypes)
                    rospy.loginfo(f"Detection class set to {class_name}")
                    return self.classes, prototypes

            class_name = input("New class detected. What would you like to call this class? \n") + " - " + barcode
            self.unset_class = class_name

    def unset_class_selection_wizard(self, annotated_img, raw_img, amount_of_prototypes, vis_result=True):
        class_ = self.unset_class
        output_directory, self.unset_class = self.roi_selector(annotated_img, raw_img, class_,
                                                               self.path_to_dataset, vis_result)
        if output_directory is not None:
            class_prototype = self._calculate_class_prototype(output_directory)
            self.prototype_dict[class_] = class_prototype
            self.classes, class_prototype_tensor = self._load_prototypes_from_dict(class_, amount_of_prototypes)
            return self.classes, class_prototype_tensor
        else:
            return None, None

    def _calculate_class_prototype(self, class_dir, batch_size=150):
        images = [image for image in class_dir.iterdir() if image.is_file()]
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
        return class_prototype

    def _fill_prototype_dict(self, path_to_dataset, batch_size=150):
        prototype_dict = {}
        if not path_to_dataset.exists():
            raise Exception("Please specify a valid path to dataset")
        for class_path in path_to_dataset.iterdir():
            print("Processing:", class_path)
            if class_path.is_dir():
                class_prototype = self._calculate_class_prototype(class_path, batch_size)
                prototype_dict[class_path.name] = class_prototype
        return prototype_dict


class PMF:
    def __init__(self, pmf_model_path: Path, image_loader: ImageLoader, classification_confidence_threshold: float,
                 path_to_dataset: Optional[Path] = None, reload_prototypes: bool = False, device="cuda:0",
                 amount_of_prototypes=5):
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
        self.amount_of_prototypes = amount_of_prototypes

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
            classes = [self.get_current_class() if score != 0 else "_" for score in scores.tolist()]
            return scores, classes

    def set_class_to_find(self, class_to_find):
        class_list, class_prototypes = self.prototype_loader.load_prototypes(class_to_find,
                                                                             self.amount_of_prototypes) or (None, None)
        if class_list is not None and class_prototypes is not None:
            self.class_list = class_list
            self.protonet.update_prototypes(class_prototypes)

    def get_current_class(self):
        return self.prototype_loader.get_current_class()

    def check_for_new_class_selection(self, annotated_img, raw_img, vis_results=True):
        class_list, class_prototypes = self.prototype_loader.unset_class_selection_wizard(annotated_img, raw_img,
                                                                                          self.amount_of_prototypes)
        if class_list is not None and class_prototypes is not None:
            self.class_list = class_list
            self.protonet.update_prototypes(class_prototypes)
            self.save_model_dict(self.pmf_path)
