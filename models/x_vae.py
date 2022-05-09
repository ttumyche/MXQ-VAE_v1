import copy
from einops import rearrange

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertModel, BertConfig, GPT2Config
from transformers.optimization import get_cosine_schedule_with_warmup

from .tst import Txt_1D_CNN
from .img_nn import Img_2D_CNN
from .gpt2 import GPT2LMHeadModel



norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def denormalize(x, mean=norm[0], std=norm[1]):
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

class MODEL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.args = args
        args.config = BertConfig.from_pretrained(args.bert_model_size)
        if args.step2 in ['none']:
            self.model = X_BERT_VQVAE(args)

        elif args.step2 in ['transformer']:
            self.model = Transformer_Prior(args)

    def forward(self, x):
        return None

    def training_step(self, batch, batch_idx):
        ori_img, img, original_input_ids, input_ids, attn_masks, caption = batch

        losses = self.model('train', 'none', batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False, return_loss=True, return_recons=True)

        for k, v in losses.items():
            k = 'tr_' + k
            self.log(k, v, on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=False)
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        ori_img, img, original_input_ids, input_ids, attn_masks, caption = batch

        losses = self.model('valid', 'none', batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False,
                               return_loss=True, return_recons=True)

        for k, v in losses.items():
            k = 'eval_' + k
            self.log(k, v, on_step=True, on_epoch=None, logger=True, prog_bar=False, sync_dist=True)
        return losses['total_loss']

    def test_step(self, batch, batch_idx):
        ori_img, img, original_input_ids, input_ids, attn_masks, caption = batch
        losses = self.model('test', 'none', batch_idx, img, original_input_ids, input_ids, attn_masks, return_logits=False, return_loss=True, return_recons=True)

        for k, v in losses.items():
            k = 'eval_' + k
            self.log(k, v, on_step=True, on_epoch=None, logger=True, sync_dist=True)
        return losses['total_loss']

    def configure_optimizers(self):
        train_loader = self.train_dataloader()
        optimizer = AdamW(self.parameters(), lr=self.args.base_lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps, weight_decay=self.args.weight_decay)
        lr_scheduler = {
            'scheduler':
                get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup_epochs * len(train_loader),
                num_training_steps=self.args.epochs * len(train_loader)),
            'interval': 'step',
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass


class X_BERT_VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_enc_dec = Img_2D_CNN(args)
        self.txt_enc_dec = Txt_1D_CNN(args)


        custom_bert_tiny_config = copy.deepcopy(args.config)
        custom_bert_tiny_config.num_hidden_layers = args.x_bert_vqvae_bert_layer
        if args.cb_dim < 128:
            custom_bert_tiny_config.num_attention_heads = 1
            custom_bert_tiny_config.intermediate_size = 512
        custom_bert_tiny_config.hidden_size = args.cb_dim

        self.BERT = BertModel(custom_bert_tiny_config)

        if args.img_size == 224:
            self.z_img = 28
            z_img = 28
        elif args.img_size == 64:
            self.z_img = 8
            z_img = 8

        if args.token_length == 64:
            self.z_txt = 8
            z_txt = 8
        elif args.token_length == 80:
            self.z_txt = 20
            z_txt = 20

        if args.img_size == 224:
            h_dim = 392
        elif args.img_size == 64:
            h_dim = 32

        self.Layer = nn.Linear(z_img ** 2 + z_txt, z_img ** 2)

        self.Layer_img = nn.Linear(z_img ** 2, z_img ** 2)

        self.Layer_txt = nn.Sequential(
            nn.Linear(z_img ** 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_txt))

        self.num_codebook = args.num_codebook
        self.codebook = nn.Embedding(args.num_codebook, args.cb_dim)

    def quantize(self, step, z):
        bsz, t, dim = z.shape
        z = rearrange(z, 'b t d -> (b t) d')

        d = z.pow(2).sum(1, keepdim=True) + \
            self.codebook.weight.pow(2).sum(1) + \
            - 2 * z @ self.codebook.weight.t()

        min_encoding_idx = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_idx).view(z.shape)

        b_min_idx = rearrange(min_encoding_idx, '(b t) -> b t', t=t)

        encodings = torch.zeros(min_encoding_idx.shape[0], self.args.num_codebook, device=z.device)
        encodings.scatter_(1, min_encoding_idx.unsqueeze(1), 1)

        # vq loss
        loss_vq = F.mse_loss(z_q, z.detach())
        # commitment loss
        loss_commit = F.mse_loss(z, z_q.detach())

        # preserve gradients.
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, '(b t) d -> b t d', b=bsz)

        return loss_vq, loss_commit, z_q, b_min_idx

    def forward(self, step, logger, batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False, return_loss=False, return_recons=False):
        logits_i = self.img_enc_dec(img, None)  # bsz, dim, h, w
        bsz, dim, h, w = logits_i.shape
        rearange_i = rearrange(logits_i, 'b d h w -> b (h w) d')

        logits_t = self.txt_enc_dec(original_input_ids, input_ids, attn_masks, None)  # bsz, dim, len
        _, _, t = logits_t.shape
        rearange_t = rearrange(logits_t, 'b d t -> b t d')

        concat_enc_input = torch.cat([rearange_i, rearange_t], dim=1)
        concat_i_t = self.BERT.encoder(concat_enc_input, output_attentions=True).last_hidden_state
        concat_i_t = concat_i_t[:, :(h*w)]

        loss_vq, loss_commit, z_q, b_min_idx = self.quantize(step, concat_i_t)

        z_q_i = rearrange(self.Layer_img(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous(), 'b (h w) d -> b d h w', h=h)
        z_q_t = self.Layer_txt(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # b seq_len, d

        recon_i = self.img_enc_dec(img, z_q_i)
        recon_loss_i_mse = F.mse_loss(recon_i, img)

        recon_loss_t, acc_t, decoded_txt, recon_t, bits_per_dim, dec_output = self.txt_enc_dec(original_input_ids, input_ids, attn_masks, z_q_t)

        total_loss = recon_loss_i_mse + recon_loss_t + self.args.beta * loss_commit + loss_vq

        losses = {'bits_per_dim': bits_per_dim, 'matching_acc': acc_t.mean(),
                  'recon_loss_i': recon_loss_i_mse.mean(),
                  'recon_loss_t': recon_loss_t.mean(),
                  'commit_loss': loss_commit.mean() * self.args.beta,
                  'embedding_loss': loss_vq.mean(),
                  'total_loss': total_loss.mean()}
        return losses


class Transformer_Prior(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.img_size == 224:
            self.z_img = 28
        elif args.img_size == 64:
            self.z_img = 8

        if args.token_length == 80:
            self.z_txt = 20
        elif args.token_length == 64:
            self.z_txt = 8


        self.model = X_BERT_VQVAE(args)
        if self.args.model_load_vq is not None:
            ckpt = torch.load(args.model_load_vq)
            new_k = list(map(lambda i: '.'.join(i.split('.')[1:]), ckpt['state_dict'].keys()))
            new_ckpt = dict(zip(new_k, ckpt['state_dict'].values()))
            self.model.load_state_dict(new_ckpt)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.sos_tok = args.num_codebook
        self.cb_w_sos = nn.Embedding(args.num_codebook + 1, args.cb_dim)
        self.cb_w_sos.weight.data[:-1, ] = self.model.codebook.weight.data

        self.step2_cb_w_sos = nn.Embedding(1 + args.num_codebook, args.step2_cb_dim)

        config = GPT2Config.from_pretrained('gpt2')
        config.add_cross_attention = False
        config.n_layer = args.decoder_num_layers
        config.n_head = args.decoder_num_attn_heads
        config.vocab_size = args.num_codebook + 1

        config.n_embd = args.step2_cb_dim

        config.n_positions = self.z_img ** 2 + 1
        config.n_ctx = self.z_img ** 2 + 1

        self.prior = GPT2LMHeadModel(config, args, None, self.step2_cb_w_sos, None)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, step, logger, batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False, return_loss=False, return_recons=False):
        with torch.no_grad():
            logits_i = self.model.img_enc_dec(img, None)  # bsz, dim, h, w

            bsz, dim, h, w = logits_i.shape
            rearange_i = rearrange(logits_i, 'b d h w -> b (h w) d')

            logits_t = self.model.txt_enc_dec(original_input_ids, input_ids, attn_masks, None)  # bsz, dim, len

            _, _, t = logits_t.shape
            rearange_t = rearrange(logits_t, 'b d t -> b t d')

            concat_i_t = self.model.BERT.encoder(torch.cat([rearange_i, rearange_t], dim=1), output_attentions=True).last_hidden_state
            concat_i_t = concat_i_t[:, :(h*w)]

            _, _, z_q, b_min_idx = self.model.quantize(step, concat_i_t)

        b_min_idx_ = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=z_q.device) + self.sos_tok, b_min_idx], 1)

        dec_output = self.prior(input_ids=b_min_idx_, labels=b_min_idx_.detach(), output_attentions=True)
        loss = dec_output.loss

        prediction_scores = dec_output.logits[:, :-1, :].contiguous()
        pred_idx = torch.argmax(self.softmax(prediction_scores), dim=-1)
        acc = (pred_idx == b_min_idx).float().mean().item() * 100

        losses = {'total_loss': loss.mean(), 'step2_code_matching_acc': acc}
        return losses
