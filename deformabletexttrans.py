import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import DeformConv2d



class TextRCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Add a sequence dimension of size 1 to the input
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # Concatenate the last time step's output from both directions
        x = torch.cat((lstm_out[:, -1, :self.hidden_size], lstm_out[:, 0, self.hidden_size:]), dim=1)
        output = self.fc(x)
        return output

class PatchEmbedding(nn.Module):
      def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

      def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model's expected size ({self.image_size[0]}*{self.image_size[1]})."
        x = self.projection(x).flatten(2).transpose(1, 2)  # B, P*P, E
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class DeformableMLPHead(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, text_rcnn_hidden_size):
        super(DeformableMLPHead, self).__init__()
        self.deformable_conv1 = DeformConv2d(in_features, hidden_features, kernel_size=3, padding=1)
        self.deformable_conv2 = DeformConv2d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_features, num_classes)
        self.text_rcnn = TextRCNN(hidden_features, text_rcnn_hidden_size, num_classes)

    def forward(self, x, offset):
        x = F.relu(self.deformable_conv1(x, offset))
        x = F.relu(self.deformable_conv2(x, offset))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # MLP head for bounding box prediction
        bb_predictions = self.fc(x)

        # TextRCNN head for text recognition
        text_predictions = self.text_rcnn(x)

        return bb_predictions, text_predictions


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio, text_rcnn_hidden_size=128):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio)),
            depth
        )
        
        self.offset_layer = nn.Conv2d(embed_dim, 18, kernel_size=3, padding=1)  # 18 for 2D offsets (2*kernel_size^2)
        self.deformable_mlp_head = DeformableMLPHead(embed_dim, 256, num_classes, text_rcnn_hidden_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize transformer weights
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize Deformable Convolution weights
        for name, param in self.named_parameters():
            if "offset_layer" in name:
                nn.init.normal_(param, std=0.02)
            elif "deformable_mlp_head" in name:
                if "weight" in name:
                    nn.init.normal_(param, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        # Extract cls_token representation
        x = x[:, 0]

        # Reshape x to (B, C, H, W) before passing to the offset_layer
        x = x.view(B, -1, 1, 1)

        # Compute offsets for deformable convolution
        offset = self.offset_layer(x)

        # Predict bounding box coordinates using deformable CNN layers
        bb_predictions, text_predictions = self.deformable_mlp_head(x, offset)

        return bb_predictions, text_predictions
