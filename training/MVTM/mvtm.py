import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import spearman_corrcoef as spearman
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaForMaskedLM, RobertaConfig
from einops import repeat, rearrange
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def load_config(config_path):
    return OmegaConf.load(config_path)

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

class IF_MVTM(pl.LightningModule):
    def __init__(
        self,
        lr,
        weight_decay,
        num_channels,
        num_layers,
        num_heads,
        latent_dim,
        num_codes
    ):
        super().__init__() 
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_channels = num_channels
        self.vq_dim = 4
        self.vq_f_dim=256
        
        vqvae_args = dict(num_codes=num_codes, num_channels=num_channels)
        config_aced = load_config("/home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/configs/custom_vqgan.yaml")
        model_aced = load_vqgan(config_aced, ckpt_path="/home/groups/ChangLab/simsz/latent-diffusion/src/taming-transformers/vqgan-aced-ckpts/last.ckpt")
        self.tokenizer = model_aced
        
        self.mask_id = num_codes + 3
        
        config = RobertaConfig(
            vocab_size=num_codes+4,
            type_vocab_size=num_channels,
            max_position_embeddings=self.vq_dim**2,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=latent_dim*4
        )
        self.mvtm = RobertaForMaskedLM(config)
        
        
    def tokenize(self, x):
        z, _, [_, _, indices] = self.tokenizer.encode(x)
        return indices + 3
    
    
    def detokenize(self, indices, batch_size):
        indices -= 3
        z = self.tokenizer.quantize.get_codebook_entry(
            indices,
            shape=(batch_size*self.num_channels, self.vq_dim, self.vq_dim, self.vq_f_dim)
        )
        z = z.reshape([batch_size*self.num_channels, self.vq_f_dim, self.vq_dim, self.vq_dim])
        return self.tokenizer.decode(z)
    
    def decode(self, tokens, labels, logits, batch_size):
        scores = F.softmax(logits, dim=2)
        mask_locations = torch.where(labels != -100)
        tokens[mask_locations] = logits[mask_locations].argmax(dim=1)
        output = rearrange(tokens, 'b (c h w) -> (b c) h w', c=self.num_channels, h=self.vq_dim)
        detok = self.detokenize(output, batch_size)
        detok = rearrange(detok, '(b c) 1 h w -> b c h w', c=self.num_channels)
        return detok
        
        
    def forward(self, x, masked_ch_idx=None):
        batch_size = x.shape[0]
        device = x.device
        with torch.no_grad():
            x = rearrange(x, 'b c h w -> (b c) 1 h w')
            token_ids = self.tokenize(x)
            token_ids = token_ids.reshape(batch_size,self.num_channels * (self.vq_dim**2))
        
        input_ids, labels = self.mask_channels(token_ids.clone(), masked_ch_idx=masked_ch_idx)
        
        type_ids = torch.cat([torch.ones(self.vq_dim**2, device=device)*i for i in range(self.num_channels)]).long()
        position_ids = torch.cat([torch.arange(self.vq_dim**2, device=device) for _ in range(self.num_channels)]).long()

        out = self.mvtm(input_ids=input_ids, token_type_ids=type_ids, position_ids=position_ids, labels=labels)
        
        return out, token_ids, labels
                       
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        def linear_lr_schedule(epoch):
            return 1 - epoch / self.trainer.max_epochs

        scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_schedule)
        return [optimizer], [scheduler]
  

    def mask_channels(self, token_ids, masked_ch_idx=None):
        if masked_ch_idx is not None:
            for channel_i in masked_ch_idx:
                token_ids[:, self.vq_dim**2*channel_i:self.vq_dim**2*(channel_i + 1)] = self.mask_id
            labels = token_ids.clone()
            labels[token_ids != self.mask_id] = -100
        else:
            seq_length = (self.vq_dim**2) * self.num_channels
            num_masked = np.random.randint(1, self.num_channels - 1)
            rand_indices = torch.rand(token_ids.shape[0], self.num_channels, device=self.device).argsort(dim = -1)
            masked_indices = rand_indices[:, :num_masked]
            # Create an empty mask of shape (batch_size, input_length)
            mask = torch.zeros((token_ids.shape[0], seq_length), dtype=torch.bool, device=self.device)
            # Compute the start positions for each masked segment
            start_positions = masked_indices * self.vq_dim**2
            # Create an offset tensor to cover all 16 positions for each start position
            offsets = torch.arange(self.vq_dim**2, device=self.device).unsqueeze(0).unsqueeze(0)
            # Expand start positions and offsets to match the required dimensions
            expanded_start_positions = start_positions.unsqueeze(-1) + offsets
            # Flatten the indices to update the mask in one operation
            expanded_start_positions = expanded_start_positions.view(token_ids.shape[0], -1)
            # Set the corresponding positions in the mask to True
            mask.scatter_(1, expanded_start_positions, True)
        
            labels = token_ids.clone()
            labels[~mask] = -100
            token_ids[mask] = self.mask_id

        return token_ids, labels
        
        
    def training_step(self, train_batch, batch_idx):
        gt, mask, meta = train_batch
        out,_,_ = self(gt)
        self.log('loss', out['loss'], sync_dist=True)
        
        return out['loss'] 
        
             
    def validation_step(self, val_batch,  val_idx):
        gt, mask, meta = val_batch
        out, token_ids, labels = self(gt)

        pred = self.decode(token_ids, labels, out['logits'], batch_size=gt.shape[0])
        
        mask = mask.bool()
        mask = repeat(mask,'b h w -> b c h w', c=pred.shape[1])
        mints = (gt * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        pmints = (pred * mask).sum(dim=(2,3)) / mask.sum(dim=(2,3))
        if mints.shape[1] == 1:
            spearman_corr = spearman(pmints.squeeze(), mints.squeeze())
            sperman_corr = spearman_corr.unsqueeze(0)
        else:
            spearman_corr = spearman(pmints, mints) 
        spearman_corr = spearman_corr.mean()

        ssim_score = ssim(gt, pred)
        self.logger.log_image(key="Val reconstruction", 
                       images=[pred[0][0].cpu().detach().numpy(),gt[0][0].cpu().detach().numpy()], 
                       caption=["pred", "gt"])
        
        self.log('val_loss', out['loss'], sync_dist=True)
        self.log('val_ssim', ssim_score, sync_dist=True)
        self.log('val_spearman', spearman_corr, sync_dist=True)
