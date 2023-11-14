import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        # backbone
        self.backbone = backbone

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim() - 1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim() - 1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2)  # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        cls_scores = self.sigmoid(cls_scores)
        return cls_scores

    def predict_from_prototypes(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1  # NOTE: assume B==1

        B, nSupp, C, H, W = supp_x.shape
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2)  # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True)  # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x.view(-1, C, H, W))
        feat = feat.view(B, x.shape[1], -1)  # B, nQry, d

        predictions = self.cos_classifier(prototypes, feat)  # B, nQry, nC
        return predictions

    @staticmethod
    def _map_class_labels(values_to_map):
        """
        Map class labels to 1-k values
        """
        unique_values = torch.unique(values_to_map)
        mapping_tensor = torch.stack((unique_values, torch.arange(1, len(unique_values) + 1,
                                                                  device=unique_values.device)), dim=1)
        mask = values_to_map == mapping_tensor[:, :1]
        mapped_values = (1 - mask.sum(dim=0)) * values_to_map + (mask * mapping_tensor[:, 1:]).sum(dim=0)
        return mapped_values

    # Assume support class label is standard,
    # Assume y_class is standard
    def get_k_closest(self, class_support_images, support_images, support_labels, class_x, x_rest, y_rest, k=5):
        predictions = torch.mean(torch.nan_to_num(
            self.predict_from_prototypes(support_images, support_labels, class_support_images)), dim=1)
        _, closest_classes = torch.topk(predictions, k, dim=1)

        support_filter = ((closest_classes.view(-1, 1) - support_labels.view(-1)).transpose(-1, -2) == 0).sum(
            dim=-1).view(support_labels.shape) != 0
        B, nSupp, C, H, W = class_support_images.shape
        support_class_label = torch.zeros((B, nSupp), device=class_support_images.device, dtype=torch.int64)
        support_tensor = torch.cat(
            (class_support_images, support_images[support_filter].view(B, k * nSupp, C, H, W)), dim=1)
        support_labels = torch.cat(
            (support_class_label, self._map_class_labels(support_labels[support_filter]).view(B, k * nSupp)),
            dim=1)

        query_filter = ((closest_classes.view(-1, 1) - y_rest.view(-1)).transpose(-1, -2) == 0).sum(dim=-1).view(
            y_rest.shape) != 0
        B, nQuery, C, H, W = class_x.shape
        y_class = torch.zeros((B, nQuery), device=class_support_images.device, dtype=torch.int64)
        x = torch.cat((class_x, x_rest[query_filter].view(B, k * nQuery, C, H, W)), dim=1)
        y = torch.cat((y_class, self._map_class_labels(y_rest[query_filter]).view(B, k * nQuery)), dim=1)
        y = F.one_hot(y, num_classes=k + 1)

        for i in range(B):
            permSupport = torch.randperm(support_tensor.shape[1])
            permQuery = torch.randperm(x.shape[1])
            support_tensor[i, :] = support_tensor[i, permSupport]
            support_labels[i, :] = support_labels[i, permSupport]
            x[i, :] = x[i, permQuery]
            y[i, :] = y[i, permQuery]

        return support_tensor, support_labels, x, y

    def forward(self, SupportX, SupportY, x_rest):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        return self.predict_from_prototypes(SupportX, SupportY, x_rest)
