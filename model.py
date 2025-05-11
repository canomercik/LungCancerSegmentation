import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Gates
USE_ATT = True


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.use_att = USE_ATT
        self.attention_gates = nn.ModuleList() if self.use_att else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

            if self.use_att:
                self.attention_gates.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if self.use_att:
                skip_connection = self.attention_gates[idx // 2](x, skip_connection)

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def jaccard_loss(logits, targets, smooth=1e-6):
    """
    logits: raw model çıktısı, shape (N,1,H,W) veya (N,H,W)
    targets: aynı shape’te 0/1 mask
    """
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3))
    union = (probs + targets - probs*targets).sum(dim=(1,2,3))
    return 1 - ((inter + smooth) / (union + smooth)).mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1e-8):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        # Focal Loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

        # Dice Loss
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return focal_loss + dice_loss

class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, smooth=1e-8):
        super().__init__()
        self.alpha = alpha  # FP cezası
        self.beta = beta  # FN cezası
        self.gamma = gamma  # Focal etkisi
        self.smooth = smooth

    def forward(self, preds, targets):
        # Focal Loss
        bce = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none')

        preds = torch.sigmoid(preds)

        # Tversky Loss
        intersection = (preds * targets).sum()
        fps = (preds * (1 - targets)).sum()
        fns = ((1 - preds) * targets).sum()
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)

        tversky_loss = 1 - tversky

        focal_loss = (1 - tversky) ** self.gamma * bce.mean()

        return tversky_loss + focal_loss


"""
class CombinedLoss(nn.Module):
    def __init__(self, tversky_focal_loss, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.tversky_focal_loss = tversky_focal_loss
        self.alpha = alpha  # Segmentasyon kaybı için ağırlık
        self.beta = beta  # Sınıflandırma kaybı için ağırlık

    def forward(self, segmentation_preds, segmentation_targets, classification_preds, classification_targets):
        # Segmentasyon kaybı
        segmentation_loss = self.tversky_focal_loss(segmentation_preds, segmentation_targets)

        # Sınıflandırma kaybı (binary cross-entropy)
        classification_loss = F.binary_cross_entropy_with_logits(classification_preds, classification_targets)

        # Toplam kayıp: segmentasyon kaybı ve sınıflandırma kaybını birleştiriyoruz
        total_loss = self.alpha * segmentation_loss + self.beta * classification_loss

        return total_loss
"""


def test():
    x = torch.randn((3, 1, 512, 512))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
