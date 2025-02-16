import os
import random
import argparse
import json
from warnings import warn
from typing import List, Dict
from pathlib import Path
from functools import partial
from textwrap import wrap
from contextlib import suppress
from statistics import mean, stdev

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange, reduce
from timm.models import create_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from datasets import build_dataset
from train import get_args_parser, adjust_config, set_seed, set_run_name, count_params

from quantizers import *
from synchronizer import *
from estimators import *
import hls4ml
import copy
from pprint import pprint

class Transformer4HLS(torch.nn.Module):
      def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, norm_first, device):
          super().__init__()
          self.d_model = d_model
          self.nhead = nhead
          self.num_encoder_layers = num_encoder_layers
          self.dim_feedforward = dim_feedforward
          self.dropout = dropout
          self.activation = activation
          self.norm_first = norm_first
          self.device = device
          self._init_transformer()

      def _init_transformer(self):
          norm = nn.LayerNorm(self.d_model)
          self.transformer_encoder = nn.TransformerEncoder(
                                        nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                                   nhead=self.nhead,
                                                                   dim_feedforward=self.dim_feedforward,
                                                                   dropout=self.dropout,
                                                                   activation=self.activation,
                                                                   norm_first=self.norm_first,
                                                                   device=self.device),
                                        self.num_encoder_layers,
                                        norm=norm
                                      )

      def forward(self, src):  
          output = self.transformer_encoder(src)
          return output
      
def load_transformer_quant_config(quant_config_path: str = "./quant_config.json",
                                  norm_quant_config_path: str = "./norm_quant_config.json",
                                  num_layers: int = 1) -> dict:
    with open(quant_config_path, 'r') as f:
        quant_config = json.load(f)
    with open(norm_quant_config_path, 'r') as f:
        norm_quant_config = json.load(f)
    transformer_quant_config = {}
    for i in range(num_layers):
        transformer_quant_config[i] = copy.deepcopy(quant_config)
    transformer_quant_config['norm'] = copy.deepcopy(norm_quant_config)
    return transformer_quant_config

def layer_estimater(quant_config):
    bram_dict = {}  
    for layer_name in quant_config.keys():
        bram_dict[layer_name] = {}
        for var_name in quant_config[layer_name].keys():
            #pprint(quant_config[layer_name][var_name])
            bram_dict[layer_name][var_name] = VivadoVariableBRAMEstimator(name=var_name,**quant_config[layer_name][var_name])

    num_ram = 0
    for layer_name in bram_dict.keys():
        for var_name in bram_dict[layer_name].keys():
            ram_est = bram_dict[layer_name][var_name]
            num_ram += ram_est.get_num_ram()
    return num_ram

def main(args):
    set_seed(args.seed)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.finetune and args.ifa_head and args.clc:
        args.setting = 'ft_clca'
    elif args.finetune and args.ifa_head:
        args.setting = 'ft_cla'
    elif args.finetune:
        args.setting = 'ft_bl'
    else:
        args.setting = 'fz_bl'

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        args = args
    )
    if args.dataset_name.lower() != "imagenet":
        model.reset_classifier(args.num_classes)
    # print(model)

    model.to(args.device)

    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

    print('Total parameters (M): ', count_params(model) / (1e6))
    print('Trainable parameters (M): ', count_params(model, trainable=True) / (1e6))

    model4hls = Transformer4HLS(d_model=192, 
                          nhead=3, 
                          num_encoder_layers=args.model_depth, 
                          dim_feedforward=768, 
                          dropout=0, 
                          activation='gelu', 
                          norm_first=True, 
                          device='cpu')

    model4hls.eval()

    for i in range(args.model_depth):
        model4hls.transformer_encoder.layers[i].self_attn.in_proj_weight    = model.blocks[i].attn.qkv.weight
        model4hls.transformer_encoder.layers[i].self_attn.in_proj_bias      = model.blocks[i].attn.qkv.bias
        model4hls.transformer_encoder.layers[i].self_attn.out_proj.weight   = model.blocks[i].attn.proj.weight
        model4hls.transformer_encoder.layers[i].self_attn.out_proj.bias     = model.blocks[i].attn.proj.bias
        model4hls.transformer_encoder.layers[i].linear1.weight              = model.blocks[i].mlp.fc1.weight
        model4hls.transformer_encoder.layers[i].linear1.bias                = model.blocks[i].mlp.fc1.bias
        model4hls.transformer_encoder.layers[i].linear2.weight              = model.blocks[i].mlp.fc2.weight
        model4hls.transformer_encoder.layers[i].linear2.bias                = model.blocks[i].mlp.fc2.bias
        model4hls.transformer_encoder.layers[i].norm1.weight                = model.blocks[i].norm1.weight
        model4hls.transformer_encoder.layers[i].norm1.bias                  = model.blocks[i].norm1.bias
        model4hls.transformer_encoder.layers[i].norm2.weight                = model.blocks[i].norm2.weight
        model4hls.transformer_encoder.layers[i].norm2.bias                  = model.blocks[i].norm2.bias
    model4hls.transformer_encoder.norm.weight   = model.norm.weight
    model4hls.transformer_encoder.norm.bias     = model.norm.bias

    for idx in range(1):
        random_tensor = torch.randn(1, 3, args.input_size, args.input_size)
        # print(random_tensor)
        with torch.no_grad():
            x = model.patch_embed(random_tensor)
            x = model._pos_embed(x)
            x = model.patch_drop(x)
            x = model.norm_pre(x)
            print('Input shape of encoders = {}'.format(x.shape))
            out = x
            out2 = x
            # out, left_token, sample_idx, compl = model.blocks[0](x)
            # out2 = model4hls.transformer_encoder.layers[0](x.permute(1, 0, 2))
            for i, blk in enumerate(model.blocks):
                out, left_token, sample_idx, compl = blk(out)
            out = model.norm(out)
            out2 = model4hls(out2.permute(1, 0, 2))
            out2 = out2.permute(1, 0, 2)
            # print(out.shape)
            # print(out2.shape)
            # print(out)
            # print(out2)
            difference = (out - out2).sum()
            print('Difference between pytorch model and model4hls = {}'.format(difference))

    transformer_quant_config = load_transformer_quant_config(num_layers=args.model_depth)
    # print('Before calibration:')
    # pprint(transformer_quant_config)
    qmodel = QTransformerEncoder([QTransformerEncoderLayer(192, 
                                                        3, 
                                                        768, 
                                                        activation='gelu', 
                                                        quant_config=transformer_quant_config[i], 
                                                        calibration=True, 
                                                        device='cpu') for i in range(args.model_depth)], 
                                args.model_depth, 
                                QLayerNorm(192, quant_config=transformer_quant_config['norm'], calibration=True, device='cpu'),
                                TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=True),
                                dtype=torch.float64)
    qmodel.transfer_weights(model4hls)
    qmodel.to(torch.device('cpu'))
    qmodel.eval()

    if args.calibration:
        images, target = next(iter(test_loader))
        with torch.no_grad():
            x = model.patch_embed(images[0:1])
            x = model._pos_embed(x)
            x = model.patch_drop(x)
            x = model.norm_pre(x)
            print(x.min())
            print(x.max())
            transformer_quant_config = calibrate_transformer(qmodel, transformer_quant_config, x.permute(1, 0, 2).type(torch.float64).to(torch.device('cpu')))
        print('save transformer_quant_config')
        torch.save(transformer_quant_config, './transformer_quant_config_{}.pth'.format(args.input_size))
    else:
        transformer_quant_config = torch.load('./transformer_quant_config_{}.pth'.format(args.input_size))

    print('After calibration:')
    pprint(transformer_quant_config)

    BRAMstate = gen_init_BRAMaware_state(num_layers=args.model_depth, 
                                         weight_bits=7, 
                                        #  weight_bits=18, 
                                         table_input_bits=12, 
                                         table_output_bits=18, 
                                         intermediate_bits=24,
                                         result_bits=18)
    DSPstate = gen_init_nonBRAMaware_state(num_layers=args.model_depth)
    state = {**BRAMstate, **DSPstate}

    config = hls4ml.utils.config_from_pytorch_model(model4hls, 
                                                granularity='name',
                                                backend='Vitis',
                                                input_shapes=[[1, (args.input_size/16)**2+1, 192]], 
                                                default_precision='ap_fixed<18,5,AP_RND_CONV,AP_SAT>', 
                                                inputs_channel_last=True, 
                                                transpose_outputs=False)

    valid = sync_quant_config(transformer_quant_config, config, state)
    print(valid)

    qmodel = QTransformerEncoder([QTransformerEncoderLayer(embed_dim=192, 
                                                           num_heads=3, 
                                                           hidden_dim=768, 
                                                           activation='gelu', 
                                                           quant_config=transformer_quant_config[i], 
                                                           calibration=False, 
                                                           device='cpu') for i in range(args.model_depth)], 
                                args.model_depth, 
                                QLayerNorm(normalized_shape=192, 
                                           quant_config=transformer_quant_config['norm'], 
                                           calibration=False, 
                                           device='cpu'),
                                TorchQuantizer(bitwidth=18, int_bitwidth=5, signed=True, calibration=False),
                                dtype=torch.float64)
    qmodel.transfer_weights(model4hls)
    qmodel.to(torch.device('cpu'))
    qmodel.eval()
    for layer_config in config['LayerName'].keys():
        if layer_config.endswith('self_attn'):
            config['LayerName'][layer_config]['TilingFactor'] = [1,1,1]
        elif layer_config.endswith('ffn'):
            config['LayerName'][layer_config]['TilingFactor'] = [1,1,12]
    hls_model = hls4ml.converters.convert_from_pytorch_model(
                                                                model4hls,
                                                                [[1, int((args.input_size/16)**2+1), 192]],
                                                                output_dir='./hls/deit_tiny_w8_Bdk-1_Bffn-12_{}'.format(args.input_size),
                                                                project_name='myproject',
                                                                backend='Vitis',
                                                                part='xcu55c-fsvh2892-2L-e',
                                                                #board='alveo-u55c',
                                                                hls_config=config,
                                                                io_type='io_tile_stream',
                                                            )
    hls_model.compile()

    images, target = next(iter(test_loader))
    with torch.no_grad():
        x = model.patch_embed(images[0:1])
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        print('Double check input shape of encoders = {}'.format(x.shape))
        print(x.shape) 
        # print(x)
        np.savetxt('hls/deit_tiny_w8_Bdk-1_Bffn-12_{}/tb_data/tb_input_features.dat'.format(args.input_size), x.numpy().flatten().reshape(1, -1), fmt="%.6f", delimiter=" ")
        output = qmodel(x.permute(1, 0, 2).type(torch.float64))
        output = output.permute(1, 0, 2)
        encoder_out2 = model4hls(x.permute(1, 0, 2))
        encoder_out2 = encoder_out2.permute(1, 0, 2)
        hls_output = hls_model.predict(x.numpy())
        np.savetxt('hls/deit_tiny_w8_Bdk-1_Bffn-12_{}/tb_data/tb_output_predictions.dat'.format(args.input_size), encoder_out2.numpy().flatten().reshape(1, -1), fmt="%.6f", delimiter=" ")
        test_output1 = hls_output - encoder_out2.flatten().numpy()
        test_output1_sum = np.sum(test_output1)
        print('Sum of difference between qmodel output and hls_output = {}'.format((output.flatten().numpy() - hls_output).sum()))
        # print(test_output1)
        print('Sum of difference between pytorch output and hls_output = {}'.format(test_output1_sum))
        # print(output)
        # print(encoder_out2)
        # print(hls_output)

    num_ram = layer_estimater(parse_hls_model(hls_model))
    print('num_ram = ', num_ram)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    adjust_config(args)
    args.model = 'evit_deit_tiny_patch16_224.fb_in1k'
    args.dataset_name = 'cotton'
    args.dataset_root_path = '../../data/cotton'
    args.folder_train = 'cotton_square_new'
    args.folder_val = 'cotton_square_new'
    args.folder_test = 'cotton_square_new'
    args.device = 'cpu'
    # args.keep_rate = [0.1]
    # args.finetune = './results_clca/cotton_evit_vit_base_patch16_224.orig_in21k_0.1_30.pth'
    # args.finetune = './results_clca/cotton_evit_vit_base_patch16_224.orig_in21k_0.1_cla_clc_30.pth'
    args.finetune = None
    # args.input_size = 64
    args.model_depth = 12
    main(args)