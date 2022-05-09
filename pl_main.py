import sys
import pickle
import random
import argparse
from glob import glob
from datetime import datetime

from utils import set_seed

# pl
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models import MODEL
from dataset import DataModule

def train(args):
    set_seed(args.seed)
    pl.seed_everything(seed=args.seed, workers=True)

    wandb_logger = WandbLogger(config=args, project='project-name', entity='entity-name', save_code=True)

    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.save_path + str(args.dataset) + '_' + str(args.now),
                                                       filename='{epoch:02d}-{tr_total_loss: .2f}',
                                                       monitor='eval_total_loss', mode='min', verbose=True,
                                                       save_top_k=int(args.epochs / args.model_save_interval),
                                                       every_n_epochs=args.model_save_interval)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    data = DataModule(args=args)

    model = MODEL(args=args)

    wandb_logger.watch(model)

    trainer = pl.Trainer(logger=wandb_logger, gpus=-1, accelerator='ddp',  # , num_nodes=1,
                         max_epochs=args.epochs, val_check_interval=1.0,
                         terminate_on_nan=True,
                         checkpoint_callback=True, resume_from_checkpoint=args.model_load,
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0, log_every_n_steps=1, flush_logs_every_n_steps=10)
    trainer.fit(model, data)

    if model.global_rank != 0:
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2", type=str, default='none')

    parser.add_argument("--txt_num_layer", type=int, default=5)

    parser.add_argument("--x_bert_vqvae_bert_layer", type=int, default=1)
    parser.add_argument("--attn_temp", type=float, default=1.0)

    parser.add_argument("--bert_embedding_load", type=bool, default=False, help='load(T), scratch(F)')
    parser.add_argument("--bert_model_size", type=str, default='google/bert_uncased_L-2_H-128_A-2',
                        choices=['bert-base-uncased',  # base
                                 'google/bert_uncased_L-4_H-512_A-8',  # small
                                 'google/bert_uncased_L-4_H-256_A-4',  # mini
                                 'google/bert_uncased_L-2_H-128_A-2'])  # tiny
    parser.add_argument("--txt_word_embed_hidden_dim", type=int, default=128)  #

    # model parameters
    # mnist: 64, 64
    # flower: 224, 80
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num_codebook", type=int, default=256)
    parser.add_argument("--cb_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--token_length", type=int, default=64)

    parser.add_argument("--subset_ratio", type=float, default=1.0, help='subset of train data')
    parser.add_argument("--valid_subset_ratio", type=float, default=1.0, help='subset of valid data')

    model_ckpt = None
    parser.add_argument("--model_load", type=str, default=model_ckpt, help='restart')

    vq_ckpt = 'path/to/step1/model'
    parser.add_argument("--model_load_vq", type=str, default=vq_ckpt, help='vq model path for step2')

    # step2 - transformer decoder
    parser.add_argument("--prior_transformer_size", type=str, default='gpt')
    parser.add_argument("--prior_decoder_load", type=bool, default=False, help='load only decoder layer (exclude embeddings, lm_head ...')
    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_attn_heads", type=int, default=8)


    parser.add_argument("--vq_cb_load", type=bool, default=False, help='load vq_cb for decoder embedding')
    parser.add_argument("--step2_cb_dim", type=int, default=512)
    parser.add_argument("--extend_embedding_size", type=int, default=None, help='Not used anymore')

    decoder_hiddensize = 768
    parser.add_argument("--decoder_hiddensize", type=int, default=decoder_hiddensize, help='used when vq_cb_load == False')
    parser.add_argument("--decoder_intermediate_size", type=int, default=4 * decoder_hiddensize)

    # config for generation
    parser.add_argument("--top_k", type=list, default=[10])
    parser.add_argument("--top_p", type=list, default=[1.0])
    parser.add_argument("--temperature", type=list, default=[1.0])
    parser.add_argument("--num_return_sequence", type=int, default=1)

    # training config
    parser.add_argument("--epochs", type=int, default='epoch', help='number of epochs')
    parser.add_argument("--base_lr", type=float, default='lr')

    parser.add_argument("--train_bsz", type=int, default=384, help="number of batch size")
    parser.add_argument("--eval_bsz", type=int, default=250, help="number of batch size")

    parser.add_argument("--beta", type=float, default=0.25)

    parser.add_argument("--num_workers", type=int, default=1, help="num of workers")

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--model_save_interval", type=int, default=2, help='')

    # MNIST dset
    parser.add_argument("--MNIST_dset", type=str, default="path/to/dset", help="MNIST dset folder")

    # misc
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    port_num = str(random.randint(1000, 9999)).zfill(4)

    parser.add_argument("--now", type=str, default=now)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--port", type=str, default=port_num, help="Port number")

    args = parser.parse_args()

    output_path = args.save_path + str(args.dataset) + '_' + str(args.now)
    os.makedirs(output_path, exist_ok=True)
    os.chmod(args.save_path, 0o777)

    train(args)
