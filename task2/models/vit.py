import torch
import torch.nn as nn
import torch.nn.functional as F

# v2 - embed=128, heads=2, expansion=4, dropout=0.2, blocks=3

class PatchEmbedding(nn.Module):
    """ Embedding for CIFAR-10 images """
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.randn(self.n_patches+1, emb_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)  # B, n_patches, emb_size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # B, 1, emb_size
        x = torch.cat((cls_tokens, x), dim=1)  # B, n_patches+1, emb_size
        x += self.position_embeddings
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size=128, n_heads=2):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        B, N, C = x.shape
        keys = self.keys(x).view(B, N, self.n_heads, C // self.n_heads)
        queries = self.queries(x).view(B, N, self.n_heads, C // self.n_heads)
        values = self.values(x).view(B, N, self.n_heads, C // self.n_heads)

        energy = torch.einsum("bnqd,bnkd->bnqk", [queries, keys]) * (1. / torch.sqrt(torch.tensor(C // self.n_heads, dtype=torch.float)))
        attention = F.softmax(energy, dim=-1)
        out = torch.einsum("bnqk,bnvd->bnqd", [attention, values]).reshape(B, N, C)
        return self.fc_out(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=128, n_heads=2, expansion=4, dropout_p=0.2):
        super().__init__()
        self.attention = MultiHeadSelfAttention(emb_size, n_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Linear(expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out

class SimplifiedViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, emb_size=128, n_heads=2, expansion=4, dropout_p=0.2, n_blocks=3):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, emb_size=emb_size)
        self.transformer_blocks = nn.Sequential(*[TransformerEncoderBlock(emb_size=emb_size, n_heads=n_heads, expansion=expansion, dropout_p=dropout_p) for _ in range(n_blocks)])
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_blocks(x)
        cls_token = x[:, 0]
        out = self.classifier(cls_token)
        return out

# Example instantiation and summary
model = SimplifiedViT()