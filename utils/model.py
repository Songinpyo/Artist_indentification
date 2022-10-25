import timm

from torch import nn
from utils.loss import AdaCos


class convnext_xlarge(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = timm.create_model(model_name=args.model, pretrained=args.pretrained, num_classes=0, drop_rate=args.drop_rate)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
        )
        for param in self.classifier.parameters():
            param.requires_grad = True

        self.reducer = nn.AdaptiveAvgPool1d(128)

        self.arc = AdaCos(in_features=128, out_features=args.num_classes, m=0.5)

    def forward(self, x, label=None):
        # x = x/255.0
        # x = transforms.functional.normalize(x, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.reducer(x)
        x = self.arc(x, label)
        return x