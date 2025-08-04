#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_feedforwardnetwork_stream.h"
#include "nnet_utils/nnet_layernorm_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_multiheadattention_stream.h"
#include "nnet_utils/nnet_stream.h"
#include "nnet_utils/nnet_pruning_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/scale17.h"
#include "weights/bias17.h"
#include "weights/in_proj_weight18.h"
#include "weights/in_proj_bias18.h"
#include "weights/out_proj_weight18.h"
#include "weights/out_proj_bias18.h"
#include "weights/mask18.h"
#include "weights/scale20.h"
#include "weights/bias20.h"
#include "weights/in_proj_weight21.h"
#include "weights/in_proj_bias21.h"
#include "weights/out_proj_weight21.h"
#include "weights/out_proj_bias21.h"
#include "weights/scale23.h"
#include "weights/bias23.h"
#include "weights/in_proj_weight24.h"
#include "weights/in_proj_bias24.h"
#include "weights/out_proj_weight24.h"
#include "weights/out_proj_bias24.h"
#include "weights/mask24.h"
#include "weights/scale26.h"
#include "weights/bias26.h"
#include "weights/in_proj_weight27.h"
#include "weights/in_proj_bias27.h"
#include "weights/out_proj_weight27.h"
#include "weights/out_proj_bias27.h"
#include "weights/scale29.h"
#include "weights/bias29.h"
#include "weights/in_proj_weight30.h"
#include "weights/in_proj_bias30.h"
#include "weights/out_proj_weight30.h"
#include "weights/out_proj_bias30.h"
#include "weights/mask30.h"
#include "weights/scale32.h"
#include "weights/bias32.h"
#include "weights/in_proj_weight33.h"
#include "weights/in_proj_bias33.h"
#include "weights/out_proj_weight33.h"
#include "weights/out_proj_bias33.h"
#include "weights/scale35.h"
#include "weights/bias35.h"
#include "weights/in_proj_weight36.h"
#include "weights/in_proj_bias36.h"
#include "weights/out_proj_weight36.h"
#include "weights/out_proj_bias36.h"
#include "weights/mask36.h"
#include "weights/scale38.h"
#include "weights/bias38.h"
#include "weights/in_proj_weight39.h"
#include "weights/in_proj_bias39.h"
#include "weights/out_proj_weight39.h"
#include "weights/out_proj_bias39.h"
#include "weights/scale41.h"
#include "weights/bias41.h"
#include "weights/in_proj_weight42.h"
#include "weights/in_proj_bias42.h"
#include "weights/out_proj_weight42.h"
#include "weights/out_proj_bias42.h"
#include "weights/mask42.h"
#include "weights/scale44.h"
#include "weights/bias44.h"
#include "weights/in_proj_weight45.h"
#include "weights/in_proj_bias45.h"
#include "weights/out_proj_weight45.h"
#include "weights/out_proj_bias45.h"
#include "weights/scale47.h"
#include "weights/bias47.h"
#include "weights/in_proj_weight48.h"
#include "weights/in_proj_bias48.h"
#include "weights/out_proj_weight48.h"
#include "weights/out_proj_bias48.h"
#include "weights/mask48.h"
#include "weights/scale50.h"
#include "weights/bias50.h"
#include "weights/in_proj_weight51.h"
#include "weights/in_proj_bias51.h"
#include "weights/out_proj_weight51.h"
#include "weights/out_proj_bias51.h"
#include "weights/scale53.h"
#include "weights/bias53.h"
#include "weights/in_proj_weight54.h"
#include "weights/in_proj_bias54.h"
#include "weights/out_proj_weight54.h"
#include "weights/out_proj_bias54.h"
#include "weights/mask54.h"
#include "weights/scale56.h"
#include "weights/bias56.h"
#include "weights/in_proj_weight57.h"
#include "weights/in_proj_bias57.h"
#include "weights/out_proj_weight57.h"
#include "weights/out_proj_bias57.h"
#include "weights/scale59.h"
#include "weights/bias59.h"
#include "weights/in_proj_weight60.h"
#include "weights/in_proj_bias60.h"
#include "weights/out_proj_weight60.h"
#include "weights/out_proj_bias60.h"
#include "weights/mask60.h"
#include "weights/scale62.h"
#include "weights/bias62.h"
#include "weights/in_proj_weight63.h"
#include "weights/in_proj_bias63.h"
#include "weights/out_proj_weight63.h"
#include "weights/out_proj_bias63.h"
#include "weights/scale65.h"
#include "weights/bias65.h"
#include "weights/in_proj_weight66.h"
#include "weights/in_proj_bias66.h"
#include "weights/out_proj_weight66.h"
#include "weights/out_proj_bias66.h"
#include "weights/mask66.h"
#include "weights/scale68.h"
#include "weights/bias68.h"
#include "weights/in_proj_weight69.h"
#include "weights/in_proj_bias69.h"
#include "weights/out_proj_weight69.h"
#include "weights/out_proj_bias69.h"
#include "weights/scale71.h"
#include "weights/bias71.h"
#include "weights/in_proj_weight72.h"
#include "weights/in_proj_bias72.h"
#include "weights/out_proj_weight72.h"
#include "weights/out_proj_bias72.h"
#include "weights/mask72.h"
#include "weights/scale74.h"
#include "weights/bias74.h"
#include "weights/in_proj_weight75.h"
#include "weights/in_proj_bias75.h"
#include "weights/out_proj_weight75.h"
#include "weights/out_proj_bias75.h"
#include "weights/scale77.h"
#include "weights/bias77.h"
#include "weights/in_proj_weight78.h"
#include "weights/in_proj_bias78.h"
#include "weights/out_proj_weight78.h"
#include "weights/out_proj_bias78.h"
#include "weights/mask78.h"
#include "weights/scale80.h"
#include "weights/bias80.h"
#include "weights/in_proj_weight81.h"
#include "weights/in_proj_bias81.h"
#include "weights/out_proj_weight81.h"
#include "weights/out_proj_bias81.h"
#include "weights/scale83.h"
#include "weights/bias83.h"
#include "weights/in_proj_weight84.h"
#include "weights/in_proj_bias84.h"
#include "weights/out_proj_weight84.h"
#include "weights/out_proj_bias84.h"
#include "weights/mask84.h"
#include "weights/scale86.h"
#include "weights/bias86.h"
#include "weights/in_proj_weight87.h"
#include "weights/in_proj_bias87.h"
#include "weights/out_proj_weight87.h"
#include "weights/out_proj_bias87.h"
#include "weights/scale4.h"
#include "weights/bias4.h"

#define tokens_after_1st_pruning 228
#define tokens_after_2nd_pruning 160
#define tokens_after_3rd_pruning 113
#define enable_pruning 1

// hls-fpga-machine-learning insert layer-config
// layers_pruning1
struct pruning_config1 : nnet::pruning_config {
    static const unsigned N = 325;
    static const unsigned keep_tokens = tokens_after_1st_pruning;
};


// layers_pruning2
struct pruning_config2 : nnet::pruning_config {
    static const unsigned N = tokens_after_1st_pruning;
    static const unsigned keep_tokens = tokens_after_2nd_pruning;
};


// layers_pruning3
struct pruning_config3 : nnet::pruning_config {
    static const unsigned N = tokens_after_2nd_pruning;
    static const unsigned keep_tokens = tokens_after_3rd_pruning;
};

// layers_0_norm1
struct config17 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_0_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_0_norm1_mean_t mean_t;
    typedef layers_0_norm1_sum_t sum_t;   
    typedef layers_0_norm1_bias_t bias_t;
    typedef layers_0_norm1_scale_t scale_t;
    typedef layers_0_norm1_var_table_t var_table_t;
    typedef layers_0_norm1_accum_t accum_t;
};

// layers_0_self_attn
struct config18 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = 325;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_0_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_0_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_0_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_0_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask18_t mask_t;
    typedef layers_0_self_attn_exp_table_t exp_table_t;
    typedef layers_0_self_attn_inv_table_t inv_table_t;
    typedef layers_0_self_attn_scale_t scale_t;
    typedef layers_0_self_attn_accum_t accum_t;
    typedef layers_0_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_0_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_0_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 128;
    static const unsigned exp_range = 8;
    
};

// layers_0_add1
struct config19 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_0_norm2
struct config20 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 2;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_0_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_0_norm2_mean_t mean_t;
    typedef layers_0_norm2_sum_t sum_t;   
    typedef layers_0_norm2_bias_t bias_t;
    typedef layers_0_norm2_scale_t scale_t;
    typedef layers_0_norm2_var_table_t var_table_t;
    typedef layers_0_norm2_accum_t accum_t;
};

// layers_0_ffn
struct config21 : nnet::ffn_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_0_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_0_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_0_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_0_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_0_ffn_hidden_t hidden_t;
    typedef layers_0_ffn_accum_t accum_t;
    typedef layers_0_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_0_add2
struct config22 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_1_norm1
struct config23 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 2;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_1_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_1_norm1_mean_t mean_t;
    typedef layers_1_norm1_sum_t sum_t;   
    typedef layers_1_norm1_bias_t bias_t;
    typedef layers_1_norm1_scale_t scale_t;
    typedef layers_1_norm1_var_table_t var_table_t;
    typedef layers_1_norm1_accum_t accum_t;
};

// layers_1_self_attn
struct config24 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = 325;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_1_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_1_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_1_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_1_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask24_t mask_t;
    typedef layers_1_self_attn_exp_table_t exp_table_t;
    typedef layers_1_self_attn_inv_table_t inv_table_t;
    typedef layers_1_self_attn_scale_t scale_t;
    typedef layers_1_self_attn_accum_t accum_t;
    typedef layers_1_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_1_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_1_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_1_add1
struct config25 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_1_norm2
struct config26 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 2;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_1_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_1_norm2_mean_t mean_t;
    typedef layers_1_norm2_sum_t sum_t;   
    typedef layers_1_norm2_bias_t bias_t;
    typedef layers_1_norm2_scale_t scale_t;
    typedef layers_1_norm2_var_table_t var_table_t;
    typedef layers_1_norm2_accum_t accum_t;
};

// layers_1_ffn
struct config27 : nnet::ffn_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_1_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_1_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_1_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_1_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_1_ffn_hidden_t hidden_t;
    typedef layers_1_ffn_accum_t accum_t;
    typedef layers_1_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_1_add2
struct config28 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_2_norm1
struct config29 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_2_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_2_norm1_mean_t mean_t;
    typedef layers_2_norm1_sum_t sum_t;   
    typedef layers_2_norm1_bias_t bias_t;
    typedef layers_2_norm1_scale_t scale_t;
    typedef layers_2_norm1_var_table_t var_table_t;
    typedef layers_2_norm1_accum_t accum_t;
};

// layers_2_self_attn
struct config30 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = 325;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_2_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_2_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_2_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_2_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask30_t mask_t;
    typedef layers_2_self_attn_exp_table_t exp_table_t;
    typedef layers_2_self_attn_inv_table_t inv_table_t;
    typedef layers_2_self_attn_scale_t scale_t;
    typedef layers_2_self_attn_accum_t accum_t;
    typedef layers_2_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_2_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_2_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 16;
    static const unsigned exp_range = 8;
    
};

// layers_2_add1
struct config31 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_2_norm2
struct config32 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_2_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_2_norm2_mean_t mean_t;
    typedef layers_2_norm2_sum_t sum_t;   
    typedef layers_2_norm2_bias_t bias_t;
    typedef layers_2_norm2_scale_t scale_t;
    typedef layers_2_norm2_var_table_t var_table_t;
    typedef layers_2_norm2_accum_t accum_t;
};

// layers_2_ffn
struct config33 : nnet::ffn_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_2_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_2_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_2_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_2_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_2_ffn_hidden_t hidden_t;
    typedef layers_2_ffn_accum_t accum_t;
    typedef layers_2_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_2_add2
struct config34 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_3_norm1
struct config35 : nnet::layernorm_config {
    static const unsigned seq_len = 325;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_3_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_3_norm1_mean_t mean_t;
    typedef layers_3_norm1_sum_t sum_t;   
    typedef layers_3_norm1_bias_t bias_t;
    typedef layers_3_norm1_scale_t scale_t;
    typedef layers_3_norm1_var_table_t var_table_t;
    typedef layers_3_norm1_accum_t accum_t;
};

// layers_3_self_attn
struct config36 : nnet::mha_config {
    static const bool enable_topk  = enable_pruning;
    static const int  topk         = tokens_after_1st_pruning-1;
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = 325;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_3_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_3_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_3_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_3_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask36_t mask_t;
    typedef layers_3_self_attn_exp_table_t exp_table_t;
    typedef layers_3_self_attn_inv_table_t inv_table_t;
    typedef layers_3_self_attn_scale_t scale_t;
    typedef layers_3_self_attn_accum_t accum_t;
    typedef layers_3_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_3_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_3_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_3_add1
struct config37 : nnet::merge_config {
    static const unsigned n_elem = 192*325*1;
};

// layers_3_norm2
struct config38 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_3_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_3_norm2_mean_t mean_t;
    typedef layers_3_norm2_sum_t sum_t;   
    typedef layers_3_norm2_bias_t bias_t;
    typedef layers_3_norm2_scale_t scale_t;
    typedef layers_3_norm2_var_table_t var_table_t;
    typedef layers_3_norm2_accum_t accum_t;
};

// layers_3_ffn
struct config39 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_3_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_3_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_3_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_3_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_3_ffn_hidden_t hidden_t;
    typedef layers_3_ffn_accum_t accum_t;
    typedef layers_3_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_3_add2
struct config40 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_4_norm1
struct config41 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_4_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_4_norm1_mean_t mean_t;
    typedef layers_4_norm1_sum_t sum_t;   
    typedef layers_4_norm1_bias_t bias_t;
    typedef layers_4_norm1_scale_t scale_t;
    typedef layers_4_norm1_var_table_t var_table_t;
    typedef layers_4_norm1_accum_t accum_t;
};

// layers_4_self_attn
struct config42 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_4_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_4_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_4_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_4_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask42_t mask_t;
    typedef layers_4_self_attn_exp_table_t exp_table_t;
    typedef layers_4_self_attn_inv_table_t inv_table_t;
    typedef layers_4_self_attn_scale_t scale_t;
    typedef layers_4_self_attn_accum_t accum_t;
    typedef layers_4_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_4_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_4_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 32;
    static const unsigned exp_range = 8;
    
};

// layers_4_add1
struct config43 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_4_norm2
struct config44 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_4_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_4_norm2_mean_t mean_t;
    typedef layers_4_norm2_sum_t sum_t;   
    typedef layers_4_norm2_bias_t bias_t;
    typedef layers_4_norm2_scale_t scale_t;
    typedef layers_4_norm2_var_table_t var_table_t;
    typedef layers_4_norm2_accum_t accum_t;
};

// layers_4_ffn
struct config45 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_4_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_4_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_4_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_4_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_4_ffn_hidden_t hidden_t;
    typedef layers_4_ffn_accum_t accum_t;
    typedef layers_4_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_4_add2
struct config46 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_5_norm1
struct config47 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_5_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_5_norm1_mean_t mean_t;
    typedef layers_5_norm1_sum_t sum_t;   
    typedef layers_5_norm1_bias_t bias_t;
    typedef layers_5_norm1_scale_t scale_t;
    typedef layers_5_norm1_var_table_t var_table_t;
    typedef layers_5_norm1_accum_t accum_t;
};

// layers_5_self_attn
struct config48 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_5_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_5_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_5_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_5_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask48_t mask_t;
    typedef layers_5_self_attn_exp_table_t exp_table_t;
    typedef layers_5_self_attn_inv_table_t inv_table_t;
    typedef layers_5_self_attn_scale_t scale_t;
    typedef layers_5_self_attn_accum_t accum_t;
    typedef layers_5_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_5_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_5_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 32;
    static const unsigned exp_range = 8;
    
};

// layers_5_add1
struct config49 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_5_norm2
struct config50 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_5_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_5_norm2_mean_t mean_t;
    typedef layers_5_norm2_sum_t sum_t;   
    typedef layers_5_norm2_bias_t bias_t;
    typedef layers_5_norm2_scale_t scale_t;
    typedef layers_5_norm2_var_table_t var_table_t;
    typedef layers_5_norm2_accum_t accum_t;
};

// layers_5_ffn
struct config51 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_5_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_5_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_5_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_5_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_5_ffn_hidden_t hidden_t;
    typedef layers_5_ffn_accum_t accum_t;
    typedef layers_5_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_5_add2
struct config52 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_6_norm1
struct config53 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_6_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_6_norm1_mean_t mean_t;
    typedef layers_6_norm1_sum_t sum_t;   
    typedef layers_6_norm1_bias_t bias_t;
    typedef layers_6_norm1_scale_t scale_t;
    typedef layers_6_norm1_var_table_t var_table_t;
    typedef layers_6_norm1_accum_t accum_t;
};

// layers_6_self_attn
struct config54 : nnet::mha_config {
    static const bool enable_topk  = enable_pruning;
    static const int  topk         = tokens_after_2nd_pruning-1;
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_1st_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_6_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_6_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_6_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_6_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask54_t mask_t;
    typedef layers_6_self_attn_exp_table_t exp_table_t;
    typedef layers_6_self_attn_inv_table_t inv_table_t;
    typedef layers_6_self_attn_scale_t scale_t;
    typedef layers_6_self_attn_accum_t accum_t;
    typedef layers_6_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_6_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_6_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_6_add1
struct config55 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_1st_pruning*1;
};

// layers_6_norm2
struct config56 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 1;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_6_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_6_norm2_mean_t mean_t;
    typedef layers_6_norm2_sum_t sum_t;   
    typedef layers_6_norm2_bias_t bias_t;
    typedef layers_6_norm2_scale_t scale_t;
    typedef layers_6_norm2_var_table_t var_table_t;
    typedef layers_6_norm2_accum_t accum_t;
};

// layers_6_ffn
struct config57 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_6_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_6_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_6_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_6_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_6_ffn_hidden_t hidden_t;
    typedef layers_6_ffn_accum_t accum_t;
    typedef layers_6_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_6_add2
struct config58 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_7_norm1
struct config59 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 2;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_7_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_7_norm1_mean_t mean_t;
    typedef layers_7_norm1_sum_t sum_t;   
    typedef layers_7_norm1_bias_t bias_t;
    typedef layers_7_norm1_scale_t scale_t;
    typedef layers_7_norm1_var_table_t var_table_t;
    typedef layers_7_norm1_accum_t accum_t;
};

// layers_7_self_attn
struct config60 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_7_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_7_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_7_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_7_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask60_t mask_t;
    typedef layers_7_self_attn_exp_table_t exp_table_t;
    typedef layers_7_self_attn_inv_table_t inv_table_t;
    typedef layers_7_self_attn_scale_t scale_t;
    typedef layers_7_self_attn_accum_t accum_t;
    typedef layers_7_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_7_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_7_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 32;
    static const unsigned exp_range = 8;
    
};

// layers_7_add1
struct config61 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_7_norm2
struct config62 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 2;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_7_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_7_norm2_mean_t mean_t;
    typedef layers_7_norm2_sum_t sum_t;   
    typedef layers_7_norm2_bias_t bias_t;
    typedef layers_7_norm2_scale_t scale_t;
    typedef layers_7_norm2_var_table_t var_table_t;
    typedef layers_7_norm2_accum_t accum_t;
};

// layers_7_ffn
struct config63 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_7_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_7_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_7_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_7_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_7_ffn_hidden_t hidden_t;
    typedef layers_7_ffn_accum_t accum_t;
    typedef layers_7_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_7_add2
struct config64 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_8_norm1
struct config65 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_8_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_8_norm1_mean_t mean_t;
    typedef layers_8_norm1_sum_t sum_t;   
    typedef layers_8_norm1_bias_t bias_t;
    typedef layers_8_norm1_scale_t scale_t;
    typedef layers_8_norm1_var_table_t var_table_t;
    typedef layers_8_norm1_accum_t accum_t;
};

// layers_8_self_attn
struct config66 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_8_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_8_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_8_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_8_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask66_t mask_t;
    typedef layers_8_self_attn_exp_table_t exp_table_t;
    typedef layers_8_self_attn_inv_table_t inv_table_t;
    typedef layers_8_self_attn_scale_t scale_t;
    typedef layers_8_self_attn_accum_t accum_t;
    typedef layers_8_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_8_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_8_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_8_add1
struct config67 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_8_norm2
struct config68 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_8_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_8_norm2_mean_t mean_t;
    typedef layers_8_norm2_sum_t sum_t;   
    typedef layers_8_norm2_bias_t bias_t;
    typedef layers_8_norm2_scale_t scale_t;
    typedef layers_8_norm2_var_table_t var_table_t;
    typedef layers_8_norm2_accum_t accum_t;
};

// layers_8_ffn
struct config69 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_8_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_8_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_8_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_8_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_8_ffn_hidden_t hidden_t;
    typedef layers_8_ffn_accum_t accum_t;
    typedef layers_8_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_8_add2
struct config70 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_9_norm1
struct config71 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_9_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_9_norm1_mean_t mean_t;
    typedef layers_9_norm1_sum_t sum_t;   
    typedef layers_9_norm1_bias_t bias_t;
    typedef layers_9_norm1_scale_t scale_t;
    typedef layers_9_norm1_var_table_t var_table_t;
    typedef layers_9_norm1_accum_t accum_t;
};

// layers_9_self_attn
struct config72 : nnet::mha_config {
    static const bool enable_topk  = enable_pruning;
    static const int  topk         = tokens_after_3rd_pruning-1;
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_2nd_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_9_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_9_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_9_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_9_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask72_t mask_t;
    typedef layers_9_self_attn_exp_table_t exp_table_t;
    typedef layers_9_self_attn_inv_table_t inv_table_t;
    typedef layers_9_self_attn_scale_t scale_t;
    typedef layers_9_self_attn_accum_t accum_t;
    typedef layers_9_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_9_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_9_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_9_add1
struct config73 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_2nd_pruning*1;
};

// layers_9_norm2
struct config74 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_9_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_9_norm2_mean_t mean_t;
    typedef layers_9_norm2_sum_t sum_t;   
    typedef layers_9_norm2_bias_t bias_t;
    typedef layers_9_norm2_scale_t scale_t;
    typedef layers_9_norm2_var_table_t var_table_t;
    typedef layers_9_norm2_accum_t accum_t;
};

// layers_9_ffn
struct config75 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_9_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_9_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_9_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_9_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_9_ffn_hidden_t hidden_t;
    typedef layers_9_ffn_accum_t accum_t;
    typedef layers_9_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_9_add2
struct config76 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_3rd_pruning*1;
};

// layers_10_norm1
struct config77 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_10_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_10_norm1_mean_t mean_t;
    typedef layers_10_norm1_sum_t sum_t;   
    typedef layers_10_norm1_bias_t bias_t;
    typedef layers_10_norm1_scale_t scale_t;
    typedef layers_10_norm1_var_table_t var_table_t;
    typedef layers_10_norm1_accum_t accum_t;
};

// layers_10_self_attn
struct config78 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_10_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_10_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_10_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_10_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask78_t mask_t;
    typedef layers_10_self_attn_exp_table_t exp_table_t;
    typedef layers_10_self_attn_inv_table_t inv_table_t;
    typedef layers_10_self_attn_scale_t scale_t;
    typedef layers_10_self_attn_accum_t accum_t;
    typedef layers_10_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_10_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_10_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 32;
    static const unsigned exp_range = 8;
    
};

// layers_10_add1
struct config79 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_3rd_pruning*1;
};

// layers_10_norm2
struct config80 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 4;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_10_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_10_norm2_mean_t mean_t;
    typedef layers_10_norm2_sum_t sum_t;   
    typedef layers_10_norm2_bias_t bias_t;
    typedef layers_10_norm2_scale_t scale_t;
    typedef layers_10_norm2_var_table_t var_table_t;
    typedef layers_10_norm2_accum_t accum_t;
};

// layers_10_ffn
struct config81 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_10_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_10_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_10_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_10_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_10_ffn_hidden_t hidden_t;
    typedef layers_10_ffn_accum_t accum_t;
    typedef layers_10_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_10_add2
struct config82 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_3rd_pruning*1;
};

// layers_11_norm1
struct config83 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 8;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_11_norm1_sum_sqr_t sum_sqr_t;
    typedef layers_11_norm1_mean_t mean_t;
    typedef layers_11_norm1_sum_t sum_t;   
    typedef layers_11_norm1_bias_t bias_t;
    typedef layers_11_norm1_scale_t scale_t;
    typedef layers_11_norm1_var_table_t var_table_t;
    typedef layers_11_norm1_accum_t accum_t;
};

// layers_11_self_attn
struct config84 : nnet::mha_config {
    static const bool enable_topk  = false;  // 不要合成 top-k
    static const unsigned n_head = 3;
    static const unsigned head_dim = 64;
    static const unsigned embed_dim = 192;
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned qkv_ram_style = nnet::block;
    static const unsigned attn_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    static const unsigned inv_table_size = 4096;
    static const unsigned exp_table_size = 4096;
    typedef layers_11_self_attn_out_proj_bias_t out_proj_bias_t;
    typedef layers_11_self_attn_out_proj_weight_t out_proj_weight_t;
    typedef layers_11_self_attn_in_proj_bias_t in_proj_bias_t;
    typedef layers_11_self_attn_in_proj_weight_t in_proj_weight_t;
    typedef mask84_t mask_t;
    typedef layers_11_self_attn_exp_table_t exp_table_t;
    typedef layers_11_self_attn_inv_table_t inv_table_t;
    typedef layers_11_self_attn_scale_t scale_t;
    typedef layers_11_self_attn_accum_t accum_t;
    typedef layers_11_self_attn_in_proj_out_t in_proj_out_t;
    typedef layers_11_self_attn_out_proj_in_t out_proj_in_t;
    typedef layers_11_self_attn_row_sum_t row_sum_t;
    static const unsigned inv_range = 64;
    static const unsigned exp_range = 8;
    
};

// layers_11_add1
struct config85 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_3rd_pruning*1;
};

// layers_11_norm2
struct config86 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 8;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef layers_11_norm2_sum_sqr_t sum_sqr_t;
    typedef layers_11_norm2_mean_t mean_t;
    typedef layers_11_norm2_sum_t sum_t;   
    typedef layers_11_norm2_bias_t bias_t;
    typedef layers_11_norm2_scale_t scale_t;
    typedef layers_11_norm2_var_table_t var_table_t;
    typedef layers_11_norm2_accum_t accum_t;
};

// layers_11_ffn
struct config87 : nnet::ffn_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned hidden_dim = 768;
    static const unsigned in_ram_style = nnet::block;
    static const unsigned out_ram_style = nnet::block;
    static const bool activation_gelu = true;
    static constexpr unsigned tiling_factor[3] = {1,1,12};
    typedef layers_11_ffn_out_proj_bias_t out_proj_bias_t;
    typedef layers_11_ffn_out_proj_weight_t out_proj_weight_t;
    typedef layers_11_ffn_in_proj_bias_t in_proj_bias_t;
    typedef layers_11_ffn_in_proj_weight_t in_proj_weight_t;
    typedef layers_11_ffn_hidden_t hidden_t;
    typedef layers_11_ffn_accum_t accum_t;
    typedef layers_11_ffn_cdf_table_t cdf_table_t;
    static const unsigned cdf_table_size = 4096;
    static const unsigned cdf_table_range = 4;
};

// layers_11_add2
struct config88 : nnet::merge_config {
    static const unsigned n_elem = 192*tokens_after_3rd_pruning*1;
};

// norm
struct config4 : nnet::layernorm_config {
    static const unsigned seq_len = tokens_after_3rd_pruning;
    static const unsigned embed_dim = 192;
    static const unsigned table_size = 4096;
    static constexpr float table_range = 8;
    static constexpr unsigned tiling_factor[3] = {1,1,1};
    typedef norm_sum_sqr_t sum_sqr_t;
    typedef norm_mean_t mean_t;
    typedef norm_sum_t sum_t;   
    typedef norm_bias_t bias_t;
    typedef norm_scale_t scale_t;
    typedef norm_var_table_t var_table_t;
    typedef norm_accum_t accum_t;
};


#endif
