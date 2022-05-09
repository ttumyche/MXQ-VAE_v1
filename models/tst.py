import numpy as np

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

class Txt_1D_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ignore_idx = self.tokenizer.vocab['[PAD]']
        self.mask_token = self.tokenizer.vocab['[MASK]']
        self.len_vocab = len(self.tokenizer.vocab)
        args.config.hidden_size = args.txt_word_embed_hidden_dim
        self.bert = BertModel(args.config)

        self.softmax = nn.LogSoftmax(dim=-1)
        channels = args.txt_word_embed_hidden_dim
        num_resnet_blocks = 0
        has_resblocks = num_resnet_blocks > 0

        codebook_dim = args.cb_dim
        enc_chans = np.linspace(channels, channels * 4, args.txt_num_layer).astype(int)  # 5
        dec_chans = list(reversed(enc_chans[1:-1]))

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]

        dec_chans = [dec_init_chan, channels * 4, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            stride = 1
            padding = 0
            if args.token_length == 80:
                filter_size = [16, 16]
            elif args.token_length == 64:
                filter_size = [15, 15]
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, filter_size[0], stride=stride, padding=padding), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, filter_size[1], stride=stride, padding=padding), nn.ReLU()))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_last_hdim = codebook_dim

        enc_layers.append(nn.Conv1d(enc_chans[-1], enc_last_hdim, 1))
        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))
        dec_layers.append(nn.Conv1d(channels, self.len_vocab, 1))

        self.txt_1d_encoder = nn.Sequential(*enc_layers)
        self.txt_1d_decoder = nn.Sequential(*dec_layers)


    def forward(self, original_input_ids, input_ids, attn_mask, z):
        if z is None:
            embedded_txt = self.bert.embeddings.word_embeddings(input_ids).transpose(1, 2).contiguous()  # [bsz, dim, txt_len]
            proj_embeds = self.txt_1d_encoder(embedded_txt)  # [bsz, #cb, reduced_txt_len]
            return proj_embeds

        else:
            dec_output = self.txt_1d_decoder(z.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # [bsz, txt_len, embedding_vocab_size]
            decoded_ids = torch.argmax(self.softmax(dec_output), dim=-1)
            decoded_txt = self.tokenizer.batch_decode(decoded_ids)

            acc = (decoded_ids[input_ids != self.ignore_idx] == input_ids[input_ids != self.ignore_idx]).float().mean()
            loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)(dec_output.view(-1, self.len_vocab), input_ids.view(-1))

            # cal. bit per dim
            nll_val = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, reduction='none')(dec_output.view(-1, self.len_vocab), input_ids.view(-1)).detach()
            nll_val = nll_val.view(input_ids.shape).sum(-1)
            seq_len = input_ids.sum(-1)
            bits_per_dim = (nll_val / seq_len)
            return loss, acc, decoded_txt, decoded_txt, bits_per_dim, dec_output