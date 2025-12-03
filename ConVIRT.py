
# It's Zhang et al 2020 paper implementation.
# Implemented it because *contrastive learning* is cool.


from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import torch


# Simplified Model Building
class ImageEncoder(nn.Module):
  def __init__(self, n_in = 2048, n_hidden = 512, n_out = 512):
    super().__init__()
    self.model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
    self.model.fc = nn.Identity() # Remove the classification head
    self.mlp = nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_out)
    )

  def forward(self, img):
    emb = self.model(img) # B, 2048
    emb = self.mlp(emb) # B, 512
    return emb

class TextEncoder(nn.Module):
  def __init__(self, n_in = 768, n_hidden = 512, n_out = 512):
    super().__init__()
    config = BertConfig.from_pretrained("bert-base-uncased")
    self.model = BertModel(config)
    self.mlp = nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_out)
    )

  def forward(self, input_ids, attention_mask):
    emb = self.model(input_ids = input_ids, attention_mask = attention_mask)  # B, num_toks, 768
    emb = emb.pooler_output # B, 768
    emb = self.mlp(emb) # B, 512
    return emb

class ConVIRT(nn.Module):
  def __init__(self):
    super().__init__()
    self.encode_img = ImageEncoder()
    self.encode_txt = TextEncoder()

  def forward(self, x):
    img_emb = self.encode_img(x['img'])
    txt_emb = self.encode_txt(x['input_ids'], x['attention_mask'])
    logits = img_emb @ txt_emb # B, B
    labels = torch.arange(len(x['img']))
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    loss = (loss1+loss2)/2.0
    return loss

model = ConVIRT()
print(f"Model params: {(sum([p.nelement() for p in model.parameters()])/1e6):.2f}M")
