from torch import nn

from lightly.models.modules import SwaVProjectionHead


class SwaV(nn.Module):
    def __init__(self, backbone, num_ftrs, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(num_ftrs, num_ftrs, out_dim)

    def forward(self, high_resolution_crops, low_resolution_crops):
        high_resolution_features = [self._subforward(x) for x in high_resolution_crops]
        low_resolution_features = [self._subforward(x) for x in low_resolution_crops]
        return high_resolution_features, low_resolution_features

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features
