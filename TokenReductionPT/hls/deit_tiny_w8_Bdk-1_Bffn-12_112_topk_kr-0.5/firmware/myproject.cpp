#include <iostream>

#include "myproject.h"
#include "parameters.h"

#define elements_after_1st_pruning 4992 // 26 * 192
#define elements_after_2nd_pruning 2688 // 14 * 192
#define elements_after_3rd_pruning 1536 // 8 * 192

void myproject(
    hls::stream<input_t> &src,
    hls::stream<result_t> &layer4_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=src,layer4_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<layers_0_norm1_scale_t, 192>(scale17, "scale17.txt");
        nnet::load_weights_from_txt<layers_0_norm1_bias_t, 192>(bias17, "bias17.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_in_proj_weight_t, 110592>(in_proj_weight18, "in_proj_weight18.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_in_proj_bias_t, 576>(in_proj_bias18, "in_proj_bias18.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_out_proj_weight_t, 36864>(out_proj_weight18, "out_proj_weight18.txt");
        nnet::load_weights_from_txt<layers_0_self_attn_out_proj_bias_t, 192>(out_proj_bias18, "out_proj_bias18.txt");
        nnet::load_weights_from_txt<mask18_t, 2500>(mask18, "mask18.txt");
        nnet::load_weights_from_txt<layers_0_norm2_scale_t, 192>(scale20, "scale20.txt");
        nnet::load_weights_from_txt<layers_0_norm2_bias_t, 192>(bias20, "bias20.txt");
        nnet::load_weights_from_txt<layers_0_ffn_in_proj_weight_t, 147456>(in_proj_weight21, "in_proj_weight21.txt");
        nnet::load_weights_from_txt<layers_0_ffn_in_proj_bias_t, 768>(in_proj_bias21, "in_proj_bias21.txt");
        nnet::load_weights_from_txt<layers_0_ffn_out_proj_weight_t, 147456>(out_proj_weight21, "out_proj_weight21.txt");
        nnet::load_weights_from_txt<layers_0_ffn_out_proj_bias_t, 192>(out_proj_bias21, "out_proj_bias21.txt");
        nnet::load_weights_from_txt<layers_1_norm1_scale_t, 192>(scale23, "scale23.txt");
        nnet::load_weights_from_txt<layers_1_norm1_bias_t, 192>(bias23, "bias23.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_in_proj_weight_t, 110592>(in_proj_weight24, "in_proj_weight24.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_in_proj_bias_t, 576>(in_proj_bias24, "in_proj_bias24.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_out_proj_weight_t, 36864>(out_proj_weight24, "out_proj_weight24.txt");
        nnet::load_weights_from_txt<layers_1_self_attn_out_proj_bias_t, 192>(out_proj_bias24, "out_proj_bias24.txt");
        nnet::load_weights_from_txt<mask24_t, 2500>(mask24, "mask24.txt");
        nnet::load_weights_from_txt<layers_1_norm2_scale_t, 192>(scale26, "scale26.txt");
        nnet::load_weights_from_txt<layers_1_norm2_bias_t, 192>(bias26, "bias26.txt");
        nnet::load_weights_from_txt<layers_1_ffn_in_proj_weight_t, 147456>(in_proj_weight27, "in_proj_weight27.txt");
        nnet::load_weights_from_txt<layers_1_ffn_in_proj_bias_t, 768>(in_proj_bias27, "in_proj_bias27.txt");
        nnet::load_weights_from_txt<layers_1_ffn_out_proj_weight_t, 147456>(out_proj_weight27, "out_proj_weight27.txt");
        nnet::load_weights_from_txt<layers_1_ffn_out_proj_bias_t, 192>(out_proj_bias27, "out_proj_bias27.txt");
        nnet::load_weights_from_txt<layers_2_norm1_scale_t, 192>(scale29, "scale29.txt");
        nnet::load_weights_from_txt<layers_2_norm1_bias_t, 192>(bias29, "bias29.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_in_proj_weight_t, 110592>(in_proj_weight30, "in_proj_weight30.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_in_proj_bias_t, 576>(in_proj_bias30, "in_proj_bias30.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_out_proj_weight_t, 36864>(out_proj_weight30, "out_proj_weight30.txt");
        nnet::load_weights_from_txt<layers_2_self_attn_out_proj_bias_t, 192>(out_proj_bias30, "out_proj_bias30.txt");
        nnet::load_weights_from_txt<mask30_t, 2500>(mask30, "mask30.txt");
        nnet::load_weights_from_txt<layers_2_norm2_scale_t, 192>(scale32, "scale32.txt");
        nnet::load_weights_from_txt<layers_2_norm2_bias_t, 192>(bias32, "bias32.txt");
        nnet::load_weights_from_txt<layers_2_ffn_in_proj_weight_t, 147456>(in_proj_weight33, "in_proj_weight33.txt");
        nnet::load_weights_from_txt<layers_2_ffn_in_proj_bias_t, 768>(in_proj_bias33, "in_proj_bias33.txt");
        nnet::load_weights_from_txt<layers_2_ffn_out_proj_weight_t, 147456>(out_proj_weight33, "out_proj_weight33.txt");
        nnet::load_weights_from_txt<layers_2_ffn_out_proj_bias_t, 192>(out_proj_bias33, "out_proj_bias33.txt");
        nnet::load_weights_from_txt<layers_3_norm1_scale_t, 192>(scale35, "scale35.txt");
        nnet::load_weights_from_txt<layers_3_norm1_bias_t, 192>(bias35, "bias35.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_in_proj_weight_t, 110592>(in_proj_weight36, "in_proj_weight36.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_in_proj_bias_t, 576>(in_proj_bias36, "in_proj_bias36.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_out_proj_weight_t, 36864>(out_proj_weight36, "out_proj_weight36.txt");
        nnet::load_weights_from_txt<layers_3_self_attn_out_proj_bias_t, 192>(out_proj_bias36, "out_proj_bias36.txt");
        nnet::load_weights_from_txt<mask36_t, 2500>(mask36, "mask36.txt");
        nnet::load_weights_from_txt<layers_3_norm2_scale_t, 192>(scale38, "scale38.txt");
        nnet::load_weights_from_txt<layers_3_norm2_bias_t, 192>(bias38, "bias38.txt");
        nnet::load_weights_from_txt<layers_3_ffn_in_proj_weight_t, 147456>(in_proj_weight39, "in_proj_weight39.txt");
        nnet::load_weights_from_txt<layers_3_ffn_in_proj_bias_t, 768>(in_proj_bias39, "in_proj_bias39.txt");
        nnet::load_weights_from_txt<layers_3_ffn_out_proj_weight_t, 147456>(out_proj_weight39, "out_proj_weight39.txt");
        nnet::load_weights_from_txt<layers_3_ffn_out_proj_bias_t, 192>(out_proj_bias39, "out_proj_bias39.txt");
        nnet::load_weights_from_txt<layers_4_norm1_scale_t, 192>(scale41, "scale41.txt");
        nnet::load_weights_from_txt<layers_4_norm1_bias_t, 192>(bias41, "bias41.txt");
        nnet::load_weights_from_txt<layers_4_self_attn_in_proj_weight_t, 110592>(in_proj_weight42, "in_proj_weight42.txt");
        nnet::load_weights_from_txt<layers_4_self_attn_in_proj_bias_t, 576>(in_proj_bias42, "in_proj_bias42.txt");
        nnet::load_weights_from_txt<layers_4_self_attn_out_proj_weight_t, 36864>(out_proj_weight42, "out_proj_weight42.txt");
        nnet::load_weights_from_txt<layers_4_self_attn_out_proj_bias_t, 192>(out_proj_bias42, "out_proj_bias42.txt");
        nnet::load_weights_from_txt<mask42_t, 2500>(mask42, "mask42.txt");
        nnet::load_weights_from_txt<layers_4_norm2_scale_t, 192>(scale44, "scale44.txt");
        nnet::load_weights_from_txt<layers_4_norm2_bias_t, 192>(bias44, "bias44.txt");
        nnet::load_weights_from_txt<layers_4_ffn_in_proj_weight_t, 147456>(in_proj_weight45, "in_proj_weight45.txt");
        nnet::load_weights_from_txt<layers_4_ffn_in_proj_bias_t, 768>(in_proj_bias45, "in_proj_bias45.txt");
        nnet::load_weights_from_txt<layers_4_ffn_out_proj_weight_t, 147456>(out_proj_weight45, "out_proj_weight45.txt");
        nnet::load_weights_from_txt<layers_4_ffn_out_proj_bias_t, 192>(out_proj_bias45, "out_proj_bias45.txt");
        nnet::load_weights_from_txt<layers_5_norm1_scale_t, 192>(scale47, "scale47.txt");
        nnet::load_weights_from_txt<layers_5_norm1_bias_t, 192>(bias47, "bias47.txt");
        nnet::load_weights_from_txt<layers_5_self_attn_in_proj_weight_t, 110592>(in_proj_weight48, "in_proj_weight48.txt");
        nnet::load_weights_from_txt<layers_5_self_attn_in_proj_bias_t, 576>(in_proj_bias48, "in_proj_bias48.txt");
        nnet::load_weights_from_txt<layers_5_self_attn_out_proj_weight_t, 36864>(out_proj_weight48, "out_proj_weight48.txt");
        nnet::load_weights_from_txt<layers_5_self_attn_out_proj_bias_t, 192>(out_proj_bias48, "out_proj_bias48.txt");
        nnet::load_weights_from_txt<mask48_t, 2500>(mask48, "mask48.txt");
        nnet::load_weights_from_txt<layers_5_norm2_scale_t, 192>(scale50, "scale50.txt");
        nnet::load_weights_from_txt<layers_5_norm2_bias_t, 192>(bias50, "bias50.txt");
        nnet::load_weights_from_txt<layers_5_ffn_in_proj_weight_t, 147456>(in_proj_weight51, "in_proj_weight51.txt");
        nnet::load_weights_from_txt<layers_5_ffn_in_proj_bias_t, 768>(in_proj_bias51, "in_proj_bias51.txt");
        nnet::load_weights_from_txt<layers_5_ffn_out_proj_weight_t, 147456>(out_proj_weight51, "out_proj_weight51.txt");
        nnet::load_weights_from_txt<layers_5_ffn_out_proj_bias_t, 192>(out_proj_bias51, "out_proj_bias51.txt");
        nnet::load_weights_from_txt<layers_6_norm1_scale_t, 192>(scale53, "scale53.txt");
        nnet::load_weights_from_txt<layers_6_norm1_bias_t, 192>(bias53, "bias53.txt");
        nnet::load_weights_from_txt<layers_6_self_attn_in_proj_weight_t, 110592>(in_proj_weight54, "in_proj_weight54.txt");
        nnet::load_weights_from_txt<layers_6_self_attn_in_proj_bias_t, 576>(in_proj_bias54, "in_proj_bias54.txt");
        nnet::load_weights_from_txt<layers_6_self_attn_out_proj_weight_t, 36864>(out_proj_weight54, "out_proj_weight54.txt");
        nnet::load_weights_from_txt<layers_6_self_attn_out_proj_bias_t, 192>(out_proj_bias54, "out_proj_bias54.txt");
        nnet::load_weights_from_txt<mask54_t, 2500>(mask54, "mask54.txt");
        nnet::load_weights_from_txt<layers_6_norm2_scale_t, 192>(scale56, "scale56.txt");
        nnet::load_weights_from_txt<layers_6_norm2_bias_t, 192>(bias56, "bias56.txt");
        nnet::load_weights_from_txt<layers_6_ffn_in_proj_weight_t, 147456>(in_proj_weight57, "in_proj_weight57.txt");
        nnet::load_weights_from_txt<layers_6_ffn_in_proj_bias_t, 768>(in_proj_bias57, "in_proj_bias57.txt");
        nnet::load_weights_from_txt<layers_6_ffn_out_proj_weight_t, 147456>(out_proj_weight57, "out_proj_weight57.txt");
        nnet::load_weights_from_txt<layers_6_ffn_out_proj_bias_t, 192>(out_proj_bias57, "out_proj_bias57.txt");
        nnet::load_weights_from_txt<layers_7_norm1_scale_t, 192>(scale59, "scale59.txt");
        nnet::load_weights_from_txt<layers_7_norm1_bias_t, 192>(bias59, "bias59.txt");
        nnet::load_weights_from_txt<layers_7_self_attn_in_proj_weight_t, 110592>(in_proj_weight60, "in_proj_weight60.txt");
        nnet::load_weights_from_txt<layers_7_self_attn_in_proj_bias_t, 576>(in_proj_bias60, "in_proj_bias60.txt");
        nnet::load_weights_from_txt<layers_7_self_attn_out_proj_weight_t, 36864>(out_proj_weight60, "out_proj_weight60.txt");
        nnet::load_weights_from_txt<layers_7_self_attn_out_proj_bias_t, 192>(out_proj_bias60, "out_proj_bias60.txt");
        nnet::load_weights_from_txt<mask60_t, 2500>(mask60, "mask60.txt");
        nnet::load_weights_from_txt<layers_7_norm2_scale_t, 192>(scale62, "scale62.txt");
        nnet::load_weights_from_txt<layers_7_norm2_bias_t, 192>(bias62, "bias62.txt");
        nnet::load_weights_from_txt<layers_7_ffn_in_proj_weight_t, 147456>(in_proj_weight63, "in_proj_weight63.txt");
        nnet::load_weights_from_txt<layers_7_ffn_in_proj_bias_t, 768>(in_proj_bias63, "in_proj_bias63.txt");
        nnet::load_weights_from_txt<layers_7_ffn_out_proj_weight_t, 147456>(out_proj_weight63, "out_proj_weight63.txt");
        nnet::load_weights_from_txt<layers_7_ffn_out_proj_bias_t, 192>(out_proj_bias63, "out_proj_bias63.txt");
        nnet::load_weights_from_txt<layers_8_norm1_scale_t, 192>(scale65, "scale65.txt");
        nnet::load_weights_from_txt<layers_8_norm1_bias_t, 192>(bias65, "bias65.txt");
        nnet::load_weights_from_txt<layers_8_self_attn_in_proj_weight_t, 110592>(in_proj_weight66, "in_proj_weight66.txt");
        nnet::load_weights_from_txt<layers_8_self_attn_in_proj_bias_t, 576>(in_proj_bias66, "in_proj_bias66.txt");
        nnet::load_weights_from_txt<layers_8_self_attn_out_proj_weight_t, 36864>(out_proj_weight66, "out_proj_weight66.txt");
        nnet::load_weights_from_txt<layers_8_self_attn_out_proj_bias_t, 192>(out_proj_bias66, "out_proj_bias66.txt");
        nnet::load_weights_from_txt<mask66_t, 2500>(mask66, "mask66.txt");
        nnet::load_weights_from_txt<layers_8_norm2_scale_t, 192>(scale68, "scale68.txt");
        nnet::load_weights_from_txt<layers_8_norm2_bias_t, 192>(bias68, "bias68.txt");
        nnet::load_weights_from_txt<layers_8_ffn_in_proj_weight_t, 147456>(in_proj_weight69, "in_proj_weight69.txt");
        nnet::load_weights_from_txt<layers_8_ffn_in_proj_bias_t, 768>(in_proj_bias69, "in_proj_bias69.txt");
        nnet::load_weights_from_txt<layers_8_ffn_out_proj_weight_t, 147456>(out_proj_weight69, "out_proj_weight69.txt");
        nnet::load_weights_from_txt<layers_8_ffn_out_proj_bias_t, 192>(out_proj_bias69, "out_proj_bias69.txt");
        nnet::load_weights_from_txt<layers_9_norm1_scale_t, 192>(scale71, "scale71.txt");
        nnet::load_weights_from_txt<layers_9_norm1_bias_t, 192>(bias71, "bias71.txt");
        nnet::load_weights_from_txt<layers_9_self_attn_in_proj_weight_t, 110592>(in_proj_weight72, "in_proj_weight72.txt");
        nnet::load_weights_from_txt<layers_9_self_attn_in_proj_bias_t, 576>(in_proj_bias72, "in_proj_bias72.txt");
        nnet::load_weights_from_txt<layers_9_self_attn_out_proj_weight_t, 36864>(out_proj_weight72, "out_proj_weight72.txt");
        nnet::load_weights_from_txt<layers_9_self_attn_out_proj_bias_t, 192>(out_proj_bias72, "out_proj_bias72.txt");
        nnet::load_weights_from_txt<mask72_t, 2500>(mask72, "mask72.txt");
        nnet::load_weights_from_txt<layers_9_norm2_scale_t, 192>(scale74, "scale74.txt");
        nnet::load_weights_from_txt<layers_9_norm2_bias_t, 192>(bias74, "bias74.txt");
        nnet::load_weights_from_txt<layers_9_ffn_in_proj_weight_t, 147456>(in_proj_weight75, "in_proj_weight75.txt");
        nnet::load_weights_from_txt<layers_9_ffn_in_proj_bias_t, 768>(in_proj_bias75, "in_proj_bias75.txt");
        nnet::load_weights_from_txt<layers_9_ffn_out_proj_weight_t, 147456>(out_proj_weight75, "out_proj_weight75.txt");
        nnet::load_weights_from_txt<layers_9_ffn_out_proj_bias_t, 192>(out_proj_bias75, "out_proj_bias75.txt");
        nnet::load_weights_from_txt<layers_10_norm1_scale_t, 192>(scale77, "scale77.txt");
        nnet::load_weights_from_txt<layers_10_norm1_bias_t, 192>(bias77, "bias77.txt");
        nnet::load_weights_from_txt<layers_10_self_attn_in_proj_weight_t, 110592>(in_proj_weight78, "in_proj_weight78.txt");
        nnet::load_weights_from_txt<layers_10_self_attn_in_proj_bias_t, 576>(in_proj_bias78, "in_proj_bias78.txt");
        nnet::load_weights_from_txt<layers_10_self_attn_out_proj_weight_t, 36864>(out_proj_weight78, "out_proj_weight78.txt");
        nnet::load_weights_from_txt<layers_10_self_attn_out_proj_bias_t, 192>(out_proj_bias78, "out_proj_bias78.txt");
        nnet::load_weights_from_txt<mask78_t, 2500>(mask78, "mask78.txt");
        nnet::load_weights_from_txt<layers_10_norm2_scale_t, 192>(scale80, "scale80.txt");
        nnet::load_weights_from_txt<layers_10_norm2_bias_t, 192>(bias80, "bias80.txt");
        nnet::load_weights_from_txt<layers_10_ffn_in_proj_weight_t, 147456>(in_proj_weight81, "in_proj_weight81.txt");
        nnet::load_weights_from_txt<layers_10_ffn_in_proj_bias_t, 768>(in_proj_bias81, "in_proj_bias81.txt");
        nnet::load_weights_from_txt<layers_10_ffn_out_proj_weight_t, 147456>(out_proj_weight81, "out_proj_weight81.txt");
        nnet::load_weights_from_txt<layers_10_ffn_out_proj_bias_t, 192>(out_proj_bias81, "out_proj_bias81.txt");
        nnet::load_weights_from_txt<layers_11_norm1_scale_t, 192>(scale83, "scale83.txt");
        nnet::load_weights_from_txt<layers_11_norm1_bias_t, 192>(bias83, "bias83.txt");
        nnet::load_weights_from_txt<layers_11_self_attn_in_proj_weight_t, 110592>(in_proj_weight84, "in_proj_weight84.txt");
        nnet::load_weights_from_txt<layers_11_self_attn_in_proj_bias_t, 576>(in_proj_bias84, "in_proj_bias84.txt");
        nnet::load_weights_from_txt<layers_11_self_attn_out_proj_weight_t, 36864>(out_proj_weight84, "out_proj_weight84.txt");
        nnet::load_weights_from_txt<layers_11_self_attn_out_proj_bias_t, 192>(out_proj_bias84, "out_proj_bias84.txt");
        nnet::load_weights_from_txt<mask84_t, 2500>(mask84, "mask84.txt");
        nnet::load_weights_from_txt<layers_11_norm2_scale_t, 192>(scale86, "scale86.txt");
        nnet::load_weights_from_txt<layers_11_norm2_bias_t, 192>(bias86, "bias86.txt");
        nnet::load_weights_from_txt<layers_11_ffn_in_proj_weight_t, 147456>(in_proj_weight87, "in_proj_weight87.txt");
        nnet::load_weights_from_txt<layers_11_ffn_in_proj_bias_t, 768>(in_proj_bias87, "in_proj_bias87.txt");
        nnet::load_weights_from_txt<layers_11_ffn_out_proj_weight_t, 147456>(out_proj_weight87, "out_proj_weight87.txt");
        nnet::load_weights_from_txt<layers_11_ffn_out_proj_bias_t, 192>(out_proj_bias87, "out_proj_bias87.txt");
        nnet::load_weights_from_txt<norm_scale_t, 192>(scale4, "scale4.txt");
        nnet::load_weights_from_txt<norm_bias_t, 192>(bias4, "bias4.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<input_t> layer89_cpy1("layer89_cpy1");
    #pragma HLS BIND_STORAGE variable=layer89_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer89_cpy1 depth = 2
    hls::stream<input_t> layer89_cpy2("layer89_cpy2");
    #pragma HLS BIND_STORAGE variable=layer89_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer89_cpy2 depth = 9600
    nnet::clone_stream<input_t, input_t, 9600>(src, layer89_cpy1, layer89_cpy2); // clone_src

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS BIND_STORAGE variable=layer17_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer17_out depth = 9600
    nnet::LayerNormalize<input_t, layer17_t, config17>(layer89_cpy1, layer17_out, scale17, bias17); // layers_0_norm1

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS BIND_STORAGE variable=layer18_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer18_out depth = 1
    hls::stream<int>       layer18_topk_idx("layer18_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer18_topk_idx depth=9600
    nnet::MultiHeadAttention<layer17_t, layer18_t, config18>(layer17_out, layer18_out, layer18_topk_idx, in_proj_weight18, in_proj_bias18, out_proj_weight18, out_proj_bias18, mask18); // layers_0_self_attn

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS BIND_STORAGE variable=layer19_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer19_out depth = 2
    nnet::add<layer18_t, input_t, layer19_t, config19>(layer18_out, layer89_cpy2, layer19_out); // layers_0_add1

    hls::stream<layer19_t> layer90_cpy1("layer90_cpy1");
    #pragma HLS BIND_STORAGE variable=layer90_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer90_cpy1 depth = 2
    hls::stream<layer19_t> layer90_cpy2("layer90_cpy2");
    #pragma HLS BIND_STORAGE variable=layer90_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer90_cpy2 depth = 780
    nnet::clone_stream<layer19_t, layer19_t, 9600>(layer19_out, layer90_cpy1, layer90_cpy2); // clone_layers_0_add1

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS BIND_STORAGE variable=layer20_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer20_out depth = 192
    nnet::LayerNormalize<layer19_t, layer20_t, config20>(layer90_cpy1, layer20_out, scale20, bias20); // layers_0_norm2

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS BIND_STORAGE variable=layer21_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer21_out depth = 2
    nnet::FeedForwardNetwork<layer20_t, layer21_t, config21>(layer20_out, layer21_out, in_proj_weight21, in_proj_bias21, out_proj_weight21, out_proj_bias21); // layers_0_ffn

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS BIND_STORAGE variable=layer22_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer22_out depth = 2
    nnet::add<layer21_t, layer19_t, layer22_t, config22>(layer21_out, layer90_cpy2, layer22_out); // layers_0_add2

    hls::stream<layer22_t> layer91_cpy1("layer91_cpy1");
    #pragma HLS BIND_STORAGE variable=layer91_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer91_cpy1 depth = 2
    hls::stream<layer22_t> layer91_cpy2("layer91_cpy2");
    #pragma HLS BIND_STORAGE variable=layer91_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer91_cpy2 depth = 9600
    nnet::clone_stream<layer22_t, layer22_t, 9600>(layer22_out, layer91_cpy1, layer91_cpy2); // clone_layers_0_add2

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS BIND_STORAGE variable=layer23_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer23_out depth = 384
    nnet::LayerNormalize<layer22_t, layer23_t, config23>(layer91_cpy1, layer23_out, scale23, bias23); // layers_1_norm1

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS BIND_STORAGE variable=layer24_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer24_out depth = 1
    hls::stream<int>       layer24_topk_idx("layer24_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer24_topk_idx depth=9600
    nnet::MultiHeadAttention<layer23_t, layer24_t, config24>(layer23_out, layer24_out, layer24_topk_idx, in_proj_weight24, in_proj_bias24, out_proj_weight24, out_proj_bias24, mask24); // layers_1_self_attn

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS BIND_STORAGE variable=layer25_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer25_out depth = 2
    nnet::add<layer24_t, layer22_t, layer25_t, config25>(layer24_out, layer91_cpy2, layer25_out); // layers_1_add1

    hls::stream<layer25_t> layer92_cpy1("layer92_cpy1");
    #pragma HLS BIND_STORAGE variable=layer92_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer92_cpy1 depth = 2
    hls::stream<layer25_t> layer92_cpy2("layer92_cpy2");
    #pragma HLS BIND_STORAGE variable=layer92_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer92_cpy2 depth = 780
    nnet::clone_stream<layer25_t, layer25_t, 9600>(layer25_out, layer92_cpy1, layer92_cpy2); // clone_layers_1_add1

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS BIND_STORAGE variable=layer26_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer26_out depth = 192
    nnet::LayerNormalize<layer25_t, layer26_t, config26>(layer92_cpy1, layer26_out, scale26, bias26); // layers_1_norm2

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS BIND_STORAGE variable=layer27_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer27_out depth = 2
    nnet::FeedForwardNetwork<layer26_t, layer27_t, config27>(layer26_out, layer27_out, in_proj_weight27, in_proj_bias27, out_proj_weight27, out_proj_bias27); // layers_1_ffn

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS BIND_STORAGE variable=layer28_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer28_out depth = 2
    nnet::add<layer27_t, layer25_t, layer28_t, config28>(layer27_out, layer92_cpy2, layer28_out); // layers_1_add2

    hls::stream<layer28_t> layer93_cpy1("layer93_cpy1");
    #pragma HLS BIND_STORAGE variable=layer93_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer93_cpy1 depth = 2
    hls::stream<layer28_t> layer93_cpy2("layer93_cpy2");
    #pragma HLS BIND_STORAGE variable=layer93_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer93_cpy2 depth = 9600
    nnet::clone_stream<layer28_t, layer28_t, 9600>(layer28_out, layer93_cpy1, layer93_cpy2); // clone_layers_1_add2

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS BIND_STORAGE variable=layer29_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer29_out depth = 384
    nnet::LayerNormalize<layer28_t, layer29_t, config29>(layer93_cpy1, layer29_out, scale29, bias29); // layers_2_norm1

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS BIND_STORAGE variable=layer30_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer30_out depth = 1
    hls::stream<int>       layer30_topk_idx("layer30_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer30_topk_idx depth=9600
    nnet::MultiHeadAttention<layer29_t, layer30_t, config30>(layer29_out, layer30_out, layer30_topk_idx, in_proj_weight30, in_proj_bias30, out_proj_weight30, out_proj_bias30, mask30); // layers_2_self_attn

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS BIND_STORAGE variable=layer31_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer31_out depth = 2
    nnet::add<layer30_t, layer28_t, layer31_t, config31>(layer30_out, layer93_cpy2, layer31_out); // layers_2_add1

    hls::stream<layer31_t> layer94_cpy1("layer94_cpy1");
    #pragma HLS BIND_STORAGE variable=layer94_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer94_cpy1 depth = 2
    hls::stream<layer31_t> layer94_cpy2("layer94_cpy2");
    #pragma HLS BIND_STORAGE variable=layer94_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer94_cpy2 depth = 780
    nnet::clone_stream<layer31_t, layer31_t, 9600>(layer31_out, layer94_cpy1, layer94_cpy2); // clone_layers_2_add1

    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS BIND_STORAGE variable=layer32_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer32_out depth = 192
    nnet::LayerNormalize<layer31_t, layer32_t, config32>(layer94_cpy1, layer32_out, scale32, bias32); // layers_2_norm2

    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS BIND_STORAGE variable=layer33_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer33_out depth = 2
    nnet::FeedForwardNetwork<layer32_t, layer33_t, config33>(layer32_out, layer33_out, in_proj_weight33, in_proj_bias33, out_proj_weight33, out_proj_bias33); // layers_2_ffn

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS BIND_STORAGE variable=layer34_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer34_out depth = 2
    nnet::add<layer33_t, layer31_t, layer34_t, config34>(layer33_out, layer94_cpy2, layer34_out); // layers_2_add2

    hls::stream<layer34_t> layer95_cpy1("layer95_cpy1");
    #pragma HLS BIND_STORAGE variable=layer95_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer95_cpy1 depth = 2
    hls::stream<layer34_t> layer95_cpy2("layer95_cpy2");
    #pragma HLS BIND_STORAGE variable=layer95_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer95_cpy2 depth = 9600
    nnet::clone_stream<layer34_t, layer34_t, 9600>(layer34_out, layer95_cpy1, layer95_cpy2); // clone_layers_2_add2
    // nnet::clone_stream<input_t, layer34_t, 9600>(src, layer95_cpy1, layer95_cpy2); // clone_layers_2_add2

    hls::stream<layer35_t> layer35_out("layer35_out");
    #pragma HLS BIND_STORAGE variable=layer35_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer35_out depth = 384
    nnet::LayerNormalize<layer34_t, layer35_t, config35>(layer95_cpy1, layer35_out, scale35, bias35); // layers_3_norm1

    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS BIND_STORAGE variable=layer36_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer36_out depth = 1
    hls::stream<int>       layer36_topk_idx("layer36_topk_idx"); // 新增: Top-K index stream
    #pragma HLS BIND_STORAGE variable=layer36_topk_idx type=FIFO impl=lutram
    #pragma HLS STREAM variable=layer36_topk_idx depth=2
    nnet::MultiHeadAttention<layer35_t, layer36_t, config36>(
        layer35_out, 
        layer36_out,
        layer36_topk_idx, 
        in_proj_weight36, 
        in_proj_bias36, 
        out_proj_weight36, 
        out_proj_bias36, 
        mask36
    ); // layers_3_self_attn

    hls::stream<layer37_t> layer37_out("layer37_out");
    #pragma HLS BIND_STORAGE variable=layer37_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer37_out depth = 37272
    nnet::add<layer36_t, layer34_t, layer37_t, config37>(layer36_out, layer95_cpy2, layer37_out); // layers_3_add1

    // std::cout << "Starting 1st pruning" << std::endl;
    hls::stream<layer37_t> layer37_out_pruned("layer37_out_pruned");
    #pragma HLS BIND_STORAGE variable=layer37_out_pruned type=FIFO impl=lutram
    #pragma HLS STREAM variable=layer37_out_pruned depth=2
    nnet::PruningLayer<layer37_t, layer37_t, pruning_config1>(
        layer37_out,        // 輸入Token流
        layer37_out_pruned, // 輸出保留Token流
        // 2,            // 每2個輸入元素進行一次pruning
        layer36_topk_idx
    );
    // std::cout << "1st pruning done" << std::endl;

    hls::stream<layer37_t> layer96_cpy1("layer96_cpy1");
    #pragma HLS BIND_STORAGE variable=layer96_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer96_cpy1 depth = 2
    hls::stream<layer37_t> layer96_cpy2("layer96_cpy2");
    #pragma HLS BIND_STORAGE variable=layer96_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer96_cpy2 depth = 18433
    nnet::clone_stream<layer37_t, layer37_t, elements_after_1st_pruning>(layer37_out_pruned, layer96_cpy1, layer96_cpy2); // clone_layers_3_add1
// std::cout << "clone done" << std::endl;
    hls::stream<layer38_t> layer38_out("layer38_out");
    #pragma HLS BIND_STORAGE variable=layer38_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer38_out depth = 17856
    nnet::LayerNormalize<layer37_t, layer38_t, config38>(layer96_cpy1, layer38_out, scale38, bias38); // layers_3_norm2

    hls::stream<layer39_t> layer39_out("layer39_out");
    #pragma HLS BIND_STORAGE variable=layer39_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer39_out depth = 2
    nnet::FeedForwardNetwork<layer38_t, layer39_t, config39>(layer38_out, layer39_out, in_proj_weight39, in_proj_bias39, out_proj_weight39, out_proj_bias39); // layers_3_ffn

    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS BIND_STORAGE variable=layer40_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer40_out depth = 2
    nnet::add<layer39_t, layer37_t, layer40_t, config40>(layer39_out, layer96_cpy2, layer40_out); // layers_3_add2

    hls::stream<layer40_t> layer97_cpy1("layer97_cpy1");
    #pragma HLS BIND_STORAGE variable=layer97_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer97_cpy1 depth = 2
    hls::stream<layer40_t> layer97_cpy2("layer97_cpy2");
    #pragma HLS BIND_STORAGE variable=layer97_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer97_cpy2 depth = elements_after_1st_pruning
    nnet::clone_stream<layer40_t, layer40_t, elements_after_1st_pruning>(layer40_out, layer97_cpy1, layer97_cpy2); // clone_layers_3_add2

    hls::stream<layer41_t> layer41_out("layer41_out");
    #pragma HLS BIND_STORAGE variable=layer41_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer41_out depth = 384
    nnet::LayerNormalize<layer40_t, layer41_t, config41>(layer97_cpy1, layer41_out, scale41, bias41); // layers_4_norm1

    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS BIND_STORAGE variable=layer42_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer42_out depth = 1
    hls::stream<int>       layer42_topk_idx("layer42_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer42_topk_idx depth=elements_after_1st_pruning
    nnet::MultiHeadAttention<layer41_t, layer42_t, config42>(layer41_out, layer42_out, layer42_topk_idx, in_proj_weight42, in_proj_bias42, out_proj_weight42, out_proj_bias42, mask42); // layers_4_self_attn

    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS BIND_STORAGE variable=layer43_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer43_out depth = 2
    nnet::add<layer42_t, layer40_t, layer43_t, config43>(layer42_out, layer97_cpy2, layer43_out); // layers_4_add1

    hls::stream<layer43_t> layer98_cpy1("layer98_cpy1");
    #pragma HLS BIND_STORAGE variable=layer98_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer98_cpy1 depth = 2
    hls::stream<layer43_t> layer98_cpy2("layer98_cpy2");
    #pragma HLS BIND_STORAGE variable=layer98_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer98_cpy2 depth = 781
    nnet::clone_stream<layer43_t, layer43_t, elements_after_1st_pruning>(layer43_out, layer98_cpy1, layer98_cpy2); // clone_layers_4_add1

    hls::stream<layer44_t> layer44_out("layer44_out");
    #pragma HLS BIND_STORAGE variable=layer44_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer44_out depth = 192
    nnet::LayerNormalize<layer43_t, layer44_t, config44>(layer98_cpy1, layer44_out, scale44, bias44); // layers_4_norm2

    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS BIND_STORAGE variable=layer45_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer45_out depth = 2
    nnet::FeedForwardNetwork<layer44_t, layer45_t, config45>(layer44_out, layer45_out, in_proj_weight45, in_proj_bias45, out_proj_weight45, out_proj_bias45); // layers_4_ffn

    hls::stream<layer46_t> layer46_out("layer46_out");
    #pragma HLS BIND_STORAGE variable=layer46_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer46_out depth = 2
    nnet::add<layer45_t, layer43_t, layer46_t, config46>(layer45_out, layer98_cpy2, layer46_out); // layers_4_add2

    hls::stream<layer46_t> layer99_cpy1("layer99_cpy1");
    #pragma HLS BIND_STORAGE variable=layer99_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer99_cpy1 depth = 2
    hls::stream<layer46_t> layer99_cpy2("layer99_cpy2");
    #pragma HLS BIND_STORAGE variable=layer99_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer99_cpy2 depth = elements_after_1st_pruning
    nnet::clone_stream<layer46_t, layer46_t, elements_after_1st_pruning>(layer46_out, layer99_cpy1, layer99_cpy2); // clone_layers_4_add2

    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS BIND_STORAGE variable=layer47_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer47_out depth = 384
    nnet::LayerNormalize<layer46_t, layer47_t, config47>(layer99_cpy1, layer47_out, scale47, bias47); // layers_5_norm1

    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS BIND_STORAGE variable=layer48_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer48_out depth = 1
    hls::stream<int>       layer48_topk_idx("layer48_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer48_topk_idx depth=elements_after_1st_pruning
    nnet::MultiHeadAttention<layer47_t, layer48_t, config48>(layer47_out, layer48_out, layer48_topk_idx, in_proj_weight48, in_proj_bias48, out_proj_weight48, out_proj_bias48, mask48); // layers_5_self_attn

    hls::stream<layer49_t> layer49_out("layer49_out");
    #pragma HLS BIND_STORAGE variable=layer49_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer49_out depth = 2
    nnet::add<layer48_t, layer46_t, layer49_t, config49>(layer48_out, layer99_cpy2, layer49_out); // layers_5_add1

    hls::stream<layer49_t> layer100_cpy1("layer100_cpy1");
    #pragma HLS BIND_STORAGE variable=layer100_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer100_cpy1 depth = 2
    hls::stream<layer49_t> layer100_cpy2("layer100_cpy2");
    #pragma HLS BIND_STORAGE variable=layer100_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer100_cpy2 depth = 781
    nnet::clone_stream<layer49_t, layer49_t, elements_after_1st_pruning>(layer49_out, layer100_cpy1, layer100_cpy2); // clone_layers_5_add1

    hls::stream<layer50_t> layer50_out("layer50_out");
    #pragma HLS BIND_STORAGE variable=layer50_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer50_out depth = 192
    nnet::LayerNormalize<layer49_t, layer50_t, config50>(layer100_cpy1, layer50_out, scale50, bias50); // layers_5_norm2

    hls::stream<layer51_t> layer51_out("layer51_out");
    #pragma HLS BIND_STORAGE variable=layer51_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer51_out depth = 2
    nnet::FeedForwardNetwork<layer50_t, layer51_t, config51>(layer50_out, layer51_out, in_proj_weight51, in_proj_bias51, out_proj_weight51, out_proj_bias51); // layers_5_ffn

    hls::stream<layer52_t> layer52_out("layer52_out");
    #pragma HLS BIND_STORAGE variable=layer52_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer52_out depth = 2
    nnet::add<layer51_t, layer49_t, layer52_t, config52>(layer51_out, layer100_cpy2, layer52_out); // layers_5_add2

    hls::stream<layer52_t> layer101_cpy1("layer101_cpy1");
    #pragma HLS BIND_STORAGE variable=layer101_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer101_cpy1 depth = 2
    hls::stream<layer52_t> layer101_cpy2("layer101_cpy2");
    #pragma HLS BIND_STORAGE variable=layer101_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer101_cpy2 depth = elements_after_1st_pruning
    nnet::clone_stream<layer52_t, layer52_t, elements_after_1st_pruning>(layer52_out, layer101_cpy1, layer101_cpy2); // clone_layers_5_add2

    hls::stream<layer53_t> layer53_out("layer53_out");
    #pragma HLS BIND_STORAGE variable=layer53_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer53_out depth = 384
    nnet::LayerNormalize<layer52_t, layer53_t, config53>(layer101_cpy1, layer53_out, scale53, bias53); // layers_6_norm1

    hls::stream<layer54_t> layer54_out("layer54_out");
    #pragma HLS BIND_STORAGE variable=layer54_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer54_out depth = 1
    hls::stream<int>       layer54_topk_idx("layer54_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer54_topk_idx depth=2
    #pragma HLS BIND_STORAGE variable=layer54_topk_idx type=FIFO impl=lutram
    nnet::MultiHeadAttention<layer53_t, layer54_t, config54>(
        layer53_out, 
        layer54_out, 
        layer54_topk_idx,
        in_proj_weight54, 
        in_proj_bias54, 
        out_proj_weight54, 
        out_proj_bias54, 
        mask54
    ); // layers_6_self_attn

    hls::stream<layer55_t> layer55_out("layer55_out");
    #pragma HLS BIND_STORAGE variable=layer55_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer55_out depth = 18443
    nnet::add<layer54_t, layer52_t, layer55_t, config55>(layer54_out, layer101_cpy2, layer55_out); // layers_6_add1

    // std::cout << "Starting 2nd pruning" << std::endl;
    hls::stream<layer55_t> layer55_out_pruned("layer55_out_pruned");
    #pragma HLS BIND_STORAGE variable=layer55_out_pruned type=FIFO impl=lutram
    #pragma HLS STREAM variable=layer55_out_pruned depth=2
    nnet::PruningLayer<layer55_t, layer55_t, pruning_config2>(
        layer55_out,  // 輸入Token流
        layer55_out_pruned, // 輸出保留Token流
        // 2,            // 每2個輸入元素進行一次pruning
        layer54_topk_idx
    );
    // std::cout << "2nd pruning done" << std::endl;

    hls::stream<layer55_t> layer102_cpy1("layer102_cpy1");
    #pragma HLS BIND_STORAGE variable=layer102_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer102_cpy1 depth = 2
    hls::stream<layer55_t> layer102_cpy2("layer102_cpy2");
    #pragma HLS BIND_STORAGE variable=layer102_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer102_cpy2 depth = 9409
    nnet::clone_stream<layer55_t, layer55_t, elements_after_2nd_pruning>(layer55_out_pruned, layer102_cpy1, layer102_cpy2); // clone_layers_6_add1
// std::cout << "clone done" << std::endl;
    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS BIND_STORAGE variable=layer56_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer56_out depth = 8832
    nnet::LayerNormalize<layer55_t, layer56_t, config56>(layer102_cpy1, layer56_out, scale56, bias56); // layers_6_norm2

    hls::stream<layer57_t> layer57_out("layer57_out");
    #pragma HLS BIND_STORAGE variable=layer57_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer57_out depth = 2
    nnet::FeedForwardNetwork<layer56_t, layer57_t, config57>(layer56_out, layer57_out, in_proj_weight57, in_proj_bias57, out_proj_weight57, out_proj_bias57); // layers_6_ffn
// std::cout << "FeedForwardNetwork done" << std::endl;
    hls::stream<layer58_t> layer58_out("layer58_out");
    #pragma HLS BIND_STORAGE variable=layer58_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer58_out depth = 2
    nnet::add<layer57_t, layer55_t, layer58_t, config58>(layer57_out, layer102_cpy2, layer58_out); // layers_6_add2

    hls::stream<layer58_t> layer103_cpy1("layer103_cpy1");
    #pragma HLS BIND_STORAGE variable=layer103_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer103_cpy1 depth = 2
    hls::stream<layer58_t> layer103_cpy2("layer103_cpy2");
    #pragma HLS BIND_STORAGE variable=layer103_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer103_cpy2 depth = elements_after_2nd_pruning
    nnet::clone_stream<layer58_t, layer58_t, elements_after_2nd_pruning>(layer58_out, layer103_cpy1, layer103_cpy2); // clone_layers_6_add2

    hls::stream<layer59_t> layer59_out("layer59_out");
    #pragma HLS BIND_STORAGE variable=layer59_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer59_out depth = 384
    nnet::LayerNormalize<layer58_t, layer59_t, config59>(layer103_cpy1, layer59_out, scale59, bias59); // layers_7_norm1

    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS BIND_STORAGE variable=layer60_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer60_out depth = 1
    hls::stream<int>       layer60_topk_idx("layer60_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer60_topk_idx depth=elements_after_2nd_pruning
    nnet::MultiHeadAttention<layer59_t, layer60_t, config60>(layer59_out, layer60_out, layer60_topk_idx, in_proj_weight60, in_proj_bias60, out_proj_weight60, out_proj_bias60, mask60); // layers_7_self_attn

    hls::stream<layer61_t> layer61_out("layer61_out");
    #pragma HLS BIND_STORAGE variable=layer61_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer61_out depth = 2
    nnet::add<layer60_t, layer58_t, layer61_t, config61>(layer60_out, layer103_cpy2, layer61_out); // layers_7_add1

    hls::stream<layer61_t> layer104_cpy1("layer104_cpy1");
    #pragma HLS BIND_STORAGE variable=layer104_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer104_cpy1 depth = 2
    hls::stream<layer61_t> layer104_cpy2("layer104_cpy2");
    #pragma HLS BIND_STORAGE variable=layer104_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer104_cpy2 depth = 780
    nnet::clone_stream<layer61_t, layer61_t, elements_after_2nd_pruning>(layer61_out, layer104_cpy1, layer104_cpy2); // clone_layers_7_add1

    hls::stream<layer62_t> layer62_out("layer62_out");
    #pragma HLS BIND_STORAGE variable=layer62_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer62_out depth = 192
    nnet::LayerNormalize<layer61_t, layer62_t, config62>(layer104_cpy1, layer62_out, scale62, bias62); // layers_7_norm2

    hls::stream<layer63_t> layer63_out("layer63_out");
    #pragma HLS BIND_STORAGE variable=layer63_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer63_out depth = 2
    nnet::FeedForwardNetwork<layer62_t, layer63_t, config63>(layer62_out, layer63_out, in_proj_weight63, in_proj_bias63, out_proj_weight63, out_proj_bias63); // layers_7_ffn

    hls::stream<layer64_t> layer64_out("layer64_out");
    #pragma HLS BIND_STORAGE variable=layer64_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer64_out depth = 2
    nnet::add<layer63_t, layer61_t, layer64_t, config64>(layer63_out, layer104_cpy2, layer64_out); // layers_7_add2

    hls::stream<layer64_t> layer105_cpy1("layer105_cpy1");
    #pragma HLS BIND_STORAGE variable=layer105_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer105_cpy1 depth = 2
    hls::stream<layer64_t> layer105_cpy2("layer105_cpy2");
    #pragma HLS BIND_STORAGE variable=layer105_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer105_cpy2 depth = elements_after_2nd_pruning
    nnet::clone_stream<layer64_t, layer64_t, elements_after_2nd_pruning>(layer64_out, layer105_cpy1, layer105_cpy2); // clone_layers_7_add2

    hls::stream<layer65_t> layer65_out("layer65_out");
    #pragma HLS BIND_STORAGE variable=layer65_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer65_out depth = 384
    nnet::LayerNormalize<layer64_t, layer65_t, config65>(layer105_cpy1, layer65_out, scale65, bias65); // layers_8_norm1

    hls::stream<layer66_t> layer66_out("layer66_out");
    #pragma HLS BIND_STORAGE variable=layer66_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer66_out depth = 1
    hls::stream<int>       layer66_topk_idx("layer66_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer66_topk_idx depth=elements_after_2nd_pruning
    nnet::MultiHeadAttention<layer65_t, layer66_t, config66>(layer65_out, layer66_out, layer66_topk_idx, in_proj_weight66, in_proj_bias66, out_proj_weight66, out_proj_bias66, mask66); // layers_8_self_attn

    hls::stream<layer67_t> layer67_out("layer67_out");
    #pragma HLS BIND_STORAGE variable=layer67_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer67_out depth = 2
    nnet::add<layer66_t, layer64_t, layer67_t, config67>(layer66_out, layer105_cpy2, layer67_out); // layers_8_add1

    hls::stream<layer67_t> layer106_cpy1("layer106_cpy1");
    #pragma HLS BIND_STORAGE variable=layer106_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer106_cpy1 depth = 2
    hls::stream<layer67_t> layer106_cpy2("layer106_cpy2");
    #pragma HLS BIND_STORAGE variable=layer106_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer106_cpy2 depth = 780
    nnet::clone_stream<layer67_t, layer67_t, elements_after_2nd_pruning>(layer67_out, layer106_cpy1, layer106_cpy2); // clone_layers_8_add1

    hls::stream<layer68_t> layer68_out("layer68_out");
    #pragma HLS BIND_STORAGE variable=layer68_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer68_out depth = 192
    nnet::LayerNormalize<layer67_t, layer68_t, config68>(layer106_cpy1, layer68_out, scale68, bias68); // layers_8_norm2

    hls::stream<layer69_t> layer69_out("layer69_out");
    #pragma HLS BIND_STORAGE variable=layer69_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer69_out depth = 2
    nnet::FeedForwardNetwork<layer68_t, layer69_t, config69>(layer68_out, layer69_out, in_proj_weight69, in_proj_bias69, out_proj_weight69, out_proj_bias69); // layers_8_ffn

    hls::stream<layer70_t> layer70_out("layer70_out");
    #pragma HLS BIND_STORAGE variable=layer70_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer70_out depth = 2
    nnet::add<layer69_t, layer67_t, layer70_t, config70>(layer69_out, layer106_cpy2, layer70_out); // layers_8_add2

    hls::stream<layer70_t> layer107_cpy1("layer107_cpy1");
    #pragma HLS BIND_STORAGE variable=layer107_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer107_cpy1 depth = 2
    hls::stream<layer70_t> layer107_cpy2("layer107_cpy2");
    #pragma HLS BIND_STORAGE variable=layer107_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer107_cpy2 depth = elements_after_2nd_pruning
    nnet::clone_stream<layer70_t, layer70_t, elements_after_2nd_pruning>(layer70_out, layer107_cpy1, layer107_cpy2); // clone_layers_8_add2

    hls::stream<layer71_t> layer71_out("layer71_out");
    #pragma HLS BIND_STORAGE variable=layer71_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer71_out depth = 384
    nnet::LayerNormalize<layer70_t, layer71_t, config71>(layer107_cpy1, layer71_out, scale71, bias71); // layers_9_norm1

    hls::stream<layer72_t> layer72_out("layer72_out");
    #pragma HLS BIND_STORAGE variable=layer72_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer72_out depth = 1
    hls::stream<int>       layer72_topk_idx("layer72_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer72_topk_idx depth=2
    #pragma HLS BIND_STORAGE variable=layer72_topk_idx type=FIFO impl=lutram
    nnet::MultiHeadAttention<layer71_t, layer72_t, config72>(
        layer71_out, 
        layer72_out, 
        layer72_topk_idx,
        in_proj_weight72, 
        in_proj_bias72, 
        out_proj_weight72, 
        out_proj_bias72, 
        mask72
    ); // layers_9_self_attn

    hls::stream<layer73_t> layer73_out("layer73_out");
    #pragma HLS BIND_STORAGE variable=layer73_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer73_out depth = 4661
    nnet::add<layer72_t, layer70_t, layer73_t, config73>(layer72_out, layer107_cpy2, layer73_out); // layers_9_add1

    // std::cout << "Starting 3rd pruning" << std::endl;
    hls::stream<layer73_t> layer73_out_pruned("layer73_out_pruned");
    #pragma HLS BIND_STORAGE variable=layer73_out_pruned type=FIFO impl=lutram
    #pragma HLS STREAM variable=layer73_out_pruned depth=2
    nnet::PruningLayer<layer73_t, layer73_t, pruning_config3>(
        layer73_out,  // 輸入Token流
        layer73_out_pruned, // 輸出保留Token流
        // 2,            // 每2個輸入元素進行一次pruning
        layer72_topk_idx
    );
    // std::cout << "3rd pruning done" << std::endl;

    hls::stream<layer73_t> layer108_cpy1("layer108_cpy1");
    #pragma HLS BIND_STORAGE variable=layer108_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer108_cpy1 depth = 2
    hls::stream<layer73_t> layer108_cpy2("layer108_cpy2");
    #pragma HLS BIND_STORAGE variable=layer108_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer108_cpy2 depth = 3648
    nnet::clone_stream<layer73_t, layer73_t, elements_after_3rd_pruning>(layer73_out_pruned, layer108_cpy1, layer108_cpy2); // clone_layers_9_add1

    hls::stream<layer74_t> layer74_out("layer74_out");
    #pragma HLS BIND_STORAGE variable=layer74_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer74_out depth = 2870
    nnet::LayerNormalize<layer73_t, layer74_t, config74>(layer108_cpy1, layer74_out, scale74, bias74); // layers_9_norm2

    hls::stream<layer75_t> layer75_out("layer75_out");
    #pragma HLS BIND_STORAGE variable=layer75_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer75_out depth = 2
    nnet::FeedForwardNetwork<layer74_t, layer75_t, config75>(layer74_out, layer75_out, in_proj_weight75, in_proj_bias75, out_proj_weight75, out_proj_bias75); // layers_9_ffn

    hls::stream<layer76_t> layer76_out("layer76_out");
    #pragma HLS BIND_STORAGE variable=layer76_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer76_out depth = 2
    nnet::add<layer75_t, layer73_t, layer76_t, config76>(layer75_out, layer108_cpy2, layer76_out); // layers_9_add2

    hls::stream<layer76_t> layer109_cpy1("layer109_cpy1");
    #pragma HLS BIND_STORAGE variable=layer109_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer109_cpy1 depth = 2
    hls::stream<layer76_t> layer109_cpy2("layer109_cpy2");
    #pragma HLS BIND_STORAGE variable=layer109_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer109_cpy2 depth = elements_after_3rd_pruning
    nnet::clone_stream<layer76_t, layer76_t, elements_after_3rd_pruning>(layer76_out, layer109_cpy1, layer109_cpy2); // clone_layers_9_add2

    hls::stream<layer77_t> layer77_out("layer77_out");
    #pragma HLS BIND_STORAGE variable=layer77_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer77_out depth = 384
    nnet::LayerNormalize<layer76_t, layer77_t, config77>(layer109_cpy1, layer77_out, scale77, bias77); // layers_10_norm1

    hls::stream<layer78_t> layer78_out("layer78_out");
    #pragma HLS BIND_STORAGE variable=layer78_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer78_out depth = 1
    hls::stream<int>       layer78_topk_idx("layer78_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer78_topk_idx depth=elements_after_3rd_pruning
    nnet::MultiHeadAttention<layer77_t, layer78_t, config78>(layer77_out, layer78_out, layer78_topk_idx, in_proj_weight78, in_proj_bias78, out_proj_weight78, out_proj_bias78, mask78); // layers_10_self_attn

    hls::stream<layer79_t> layer79_out("layer79_out");
    #pragma HLS BIND_STORAGE variable=layer79_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer79_out depth = 2
    nnet::add<layer78_t, layer76_t, layer79_t, config79>(layer78_out, layer109_cpy2, layer79_out); // layers_10_add1

    hls::stream<layer79_t> layer110_cpy1("layer110_cpy1");
    #pragma HLS BIND_STORAGE variable=layer110_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer110_cpy1 depth = 2
    hls::stream<layer79_t> layer110_cpy2("layer110_cpy2");
    #pragma HLS BIND_STORAGE variable=layer110_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer110_cpy2 depth = 780
    nnet::clone_stream<layer79_t, layer79_t, elements_after_3rd_pruning>(layer79_out, layer110_cpy1, layer110_cpy2); // clone_layers_10_add1

    hls::stream<layer80_t> layer80_out("layer80_out");
    #pragma HLS BIND_STORAGE variable=layer80_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer80_out depth = 192
    nnet::LayerNormalize<layer79_t, layer80_t, config80>(layer110_cpy1, layer80_out, scale80, bias80); // layers_10_norm2

    hls::stream<layer81_t> layer81_out("layer81_out");
    #pragma HLS BIND_STORAGE variable=layer81_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer81_out depth = 2
    nnet::FeedForwardNetwork<layer80_t, layer81_t, config81>(layer80_out, layer81_out, in_proj_weight81, in_proj_bias81, out_proj_weight81, out_proj_bias81); // layers_10_ffn

    hls::stream<layer82_t> layer82_out("layer82_out");
    #pragma HLS BIND_STORAGE variable=layer82_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer82_out depth = 2
    nnet::add<layer81_t, layer79_t, layer82_t, config82>(layer81_out, layer110_cpy2, layer82_out); // layers_10_add2

    hls::stream<layer82_t> layer111_cpy1("layer111_cpy1");
    #pragma HLS BIND_STORAGE variable=layer111_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer111_cpy1 depth = 2
    hls::stream<layer82_t> layer111_cpy2("layer111_cpy2");
    #pragma HLS BIND_STORAGE variable=layer111_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer111_cpy2 depth = elements_after_3rd_pruning
    nnet::clone_stream<layer82_t, layer82_t, elements_after_3rd_pruning>(layer82_out, layer111_cpy1, layer111_cpy2); // clone_layers_10_add2

    hls::stream<layer83_t> layer83_out("layer83_out");
    #pragma HLS BIND_STORAGE variable=layer83_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer83_out depth = 384
    nnet::LayerNormalize<layer82_t, layer83_t, config83>(layer111_cpy1, layer83_out, scale83, bias83); // layers_11_norm1

    hls::stream<layer84_t> layer84_out("layer84_out");
    #pragma HLS BIND_STORAGE variable=layer84_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer84_out depth = 1
    hls::stream<int>       layer84_topk_idx("layer84_topk_idx"); // 新增: Top-K index stream
    #pragma HLS STREAM variable=layer84_topk_idx depth=elements_after_3rd_pruning
    nnet::MultiHeadAttention<layer83_t, layer84_t, config84>(layer83_out, layer84_out, layer84_topk_idx, in_proj_weight84, in_proj_bias84, out_proj_weight84, out_proj_bias84, mask84); // layers_11_self_attn

    hls::stream<layer85_t> layer85_out("layer85_out");
    #pragma HLS BIND_STORAGE variable=layer85_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer85_out depth = 2
    nnet::add<layer84_t, layer82_t, layer85_t, config85>(layer84_out, layer111_cpy2, layer85_out); // layers_11_add1

    hls::stream<layer85_t> layer112_cpy1("layer112_cpy1");
    #pragma HLS BIND_STORAGE variable=layer112_cpy1 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer112_cpy1 depth = 2
    hls::stream<layer85_t> layer112_cpy2("layer112_cpy2");
    #pragma HLS BIND_STORAGE variable=layer112_cpy2 type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer112_cpy2 depth = 780
    nnet::clone_stream<layer85_t, layer85_t, elements_after_3rd_pruning>(layer85_out, layer112_cpy1, layer112_cpy2); // clone_layers_11_add1

    hls::stream<layer86_t> layer86_out("layer86_out");
    #pragma HLS BIND_STORAGE variable=layer86_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer86_out depth = 192
    nnet::LayerNormalize<layer85_t, layer86_t, config86>(layer112_cpy1, layer86_out, scale86, bias86); // layers_11_norm2

    hls::stream<layer87_t> layer87_out("layer87_out");
    #pragma HLS BIND_STORAGE variable=layer87_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer87_out depth = 2
    nnet::FeedForwardNetwork<layer86_t, layer87_t, config87>(layer86_out, layer87_out, in_proj_weight87, in_proj_bias87, out_proj_weight87, out_proj_bias87); // layers_11_ffn

    hls::stream<layer88_t> layer88_out("layer88_out");
    #pragma HLS BIND_STORAGE variable=layer88_out type=FIFO impl=lutram
    #pragma HLS STREAM variable = layer88_out depth = 2
    nnet::add<layer87_t, layer85_t, layer88_t, config88>(layer87_out, layer112_cpy2, layer88_out); // layers_11_add2

    nnet::LayerNormalize<layer88_t, result_t, config4>(layer88_out, layer4_out, scale4, bias4); // norm

}
