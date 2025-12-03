
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
import torch.nn as nn

class ImageEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = vit_b_16(weights = ViT_B_16_Weights)

  def forward(self, x):
    # x - B, C, H, W
    x = self.model.features(x)  # B, N, C
    patch_size = x.shape[1].sqrt().int()
    x = x.transpose(1, 2) # B, C, N
    x = x.reshape(x.shape[0], x.shape[1], patch_size, patch_size) # B, C, patch_size, patch_size
    return x  

class PromptEncoder(nn.Module):
  def __init__(self, n_in = 2, n_hidden = 256, n_out = 256):
    super().__init__()
    self.mlp_points = nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, n_out)
    )
    self.mlp_boxes = nn.Sequential(
        nn.Linear(n_in*2, n_hidden),
        nn.GELU(),
        nn.Linear(n_hidden, n_out)
    )
    self.no_prompt = nn.Parameter(torch.randn(n_out))

  def forward(self, x):
    # x - points - 1point(B, num_points, 2) if the user clicks a point, 2 points(B, num_boxes, 4) if the user draws a box, 
    # None if the user didn't give anything just image is given
    if x:
      x = self.mlp_points(x) if x.shape[-1] == 2 else self.mlp_boxes(x)
    else:
      x = self.no_prompt
    return x  # B, num_points/boxes, 256 or a single 256 dim vector

class Decoder(nn.Module):
  def __init__(self, n_in = 256, n_heads = 8, n_hidden = 1024, n_layers = 6):
    super().__init__()
    layer = nn.TransformerEncoderLayer(n_in, n_heads, n_hidden, activation = 'gelu')
    self.model = nn.TransformerEncoder(layer, n_layers, norm = nn.LayerNorm(n_in))

  def forward(self, x,  n_mask_toks = 4):
    x = self.model(x) # B, img tokens(16*16=256)+prompt_tokens(lets assume a point as prompt so 1 token)+mask tokens(3 tokens)= total 260 tokens
    return x

class UpsampleHead(nn.Module):
  def __init__(self, n_out = 1, n_in = 257, n_hidden = 128):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(n_in, n_hidden, 3, padding = 1),
        nn.GELU(),
        nn.Conv2d(n_hidden, n_hidden, 3, padding = 1),
        nn.GELU(),
        nn.Conv2d(n_hidden, n_out, 1)
    )
  def forward(self, mask, img):
    x = torch.cat([mask, img], dim = 1)
    self.cnn(x) # 1, 1, H, W

class SAM(nn.Module):
  def __init__(self, n_mask_toks = 4, n_embd = 256):
    self.img_encoder = ImageEncoder()
    self.prompt_encoder = PromptEncoder()
    self.decoder = Decoder()
    self.mask_toks = nn.Parameter(torch.randn(n_mask_toks, n_embd))
    self.pointwise_conv = nn.Conv2D(768, 256, 1)
    self.upsample_heads = nn.ModuleList([UpsampleHead() for _ in range(4)])

  def forward(self, images, n_mask_toks = 4, points = None, boxes = None):
    img_embs = self.img_encoder(images) # B, 768, H, W
    img_embs_low_res = self.pointwise_conv(img_embs) # B, 256, H, W
    img_embs_low_res = img_embs_low_res.view(img_embs_low_res.shape[0], -1, img_embs_low_res.shape[1])  # B, H*W, 256

    if points or boxes:
      prompt_embs = self.prompt_encoder(points) if points else self.prompt_encoder(boxes)
    else:
      prompt_embs = self.prompt_encoder(None)
      prompt_embs = prompt_embs.unsqueeze(0).unsqueeze(0).expand(images.shape[0], -1, -1)
    
    mask_embs = self.mask_toks.unsqueeze(0).expand(images.shape[0], -1, -1)

    embs = torch.concat([img_embs_low_res, prompt_embs, mask_embs], dim = 1)
    embs = embs.transpose(0, 1) # T, B, C
    out = self.decoder(embs)  # B, n_mask_toks, 256
    out = out.transpose(0, 1) # B, T, 256

    mask_toks = out[:, -n_mask_toks: -1, :] # fecth only mask tokens
    confidence_score = out[:, -1, :]  # Confidence score

    mask_toks = mask_toks.view(-1, -1, 16, 16)  # 16 - H, W
    mask_toks = mask_toks.view(mask_toks.shape[0]*mask_toks.shape[1], 16, 16).unsqueeze(1)  # B*M, 1, H, W

    logits = []
    for i in range(mask_toks.shape[0]):
      mask = mask_toks[i:i+1]
      img = img_embs[i//3:i//3+1]

      # Upsample 1
      mask = F.interpolate(mask, mode = 'bilinear', scale_factor = 2)
      img = F.interpolate(img, mode = 'bilinear', scale_factor = 2)
      logit = self.upsample_heads[0](mask, img)

      # Upsample 2
      mask = F.interpolate(logit, mode = 'bilinear', scale_factor = 2)
      img = F.interpolate(img, mode = 'bilinear', scale_factor = 2)
      logit = self.upsample_heads[1](mask, img)

      # Upsample 3
      mask = F.interpolate(logit, mode = 'bilinear', scale_factor = 2)
      img = F.interpolate(img, mode = 'bilinear', scale_factor = 2)
      logit = self.upsample_heads[2](mask, img)

      # Upsample 4
      mask = F.interpolate(logit, mode = 'bilinear', size = (256, 256))
      img = F.interpolate(img, mode = 'bilinear', size = (256, 256))
      logit = self.upsample_heads[3](mask, img)

      logits.append(logit)

    logits = torch.cat(logits, dim = 0).view(images.shape[0], 3, -1, -1)  # B, num_masks, H, W

    # Sigmoid later will be implemented 
    # That's it :)




    


    


