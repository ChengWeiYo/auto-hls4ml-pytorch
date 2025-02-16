#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 192
#define N_INPUT_2_1 2
#define N_INPUT_1_1 192
#define N_INPUT_2_1 2
#define ln_feature_out_17 192
#define ln_seq_out_17 2
#define att_feature_out_18 192
#define att_seq_out_18 2
#define att_feature_out_18 192
#define att_seq_out_18 2
#define att_feature_out_18 192
#define att_seq_out_18 2
#define ln_feature_out_20 192
#define ln_seq_out_20 2
#define ffn_feature_out_21 192
#define ffn_seq_out_21 2
#define ffn_feature_out_21 192
#define ffn_seq_out_21 2
#define ffn_feature_out_21 192
#define ffn_seq_out_21 2
#define ln_feature_out_23 192
#define ln_seq_out_23 2
#define att_feature_out_24 192
#define att_seq_out_24 2
#define att_feature_out_24 192
#define att_seq_out_24 2
#define att_feature_out_24 192
#define att_seq_out_24 2
#define ln_feature_out_26 192
#define ln_seq_out_26 2
#define ffn_feature_out_27 192
#define ffn_seq_out_27 2
#define ffn_feature_out_27 192
#define ffn_seq_out_27 2
#define ffn_feature_out_27 192
#define ffn_seq_out_27 2
#define ln_feature_out_29 192
#define ln_seq_out_29 2
#define att_feature_out_30 192
#define att_seq_out_30 2
#define att_feature_out_30 192
#define att_seq_out_30 2
#define att_feature_out_30 192
#define att_seq_out_30 2
#define ln_feature_out_32 192
#define ln_seq_out_32 2
#define ffn_feature_out_33 192
#define ffn_seq_out_33 2
#define ffn_feature_out_33 192
#define ffn_seq_out_33 2
#define ffn_feature_out_33 192
#define ffn_seq_out_33 2
#define ln_feature_out_35 192
#define ln_seq_out_35 2
#define att_feature_out_36 192
#define att_seq_out_36 2
#define att_feature_out_36 192
#define att_seq_out_36 2
#define att_feature_out_36 192
#define att_seq_out_36 2
#define ln_feature_out_38 192
#define ln_seq_out_38 2
#define ffn_feature_out_39 192
#define ffn_seq_out_39 2
#define ffn_feature_out_39 192
#define ffn_seq_out_39 2
#define ffn_feature_out_39 192
#define ffn_seq_out_39 2
#define ln_feature_out_41 192
#define ln_seq_out_41 2
#define att_feature_out_42 192
#define att_seq_out_42 2
#define att_feature_out_42 192
#define att_seq_out_42 2
#define att_feature_out_42 192
#define att_seq_out_42 2
#define ln_feature_out_44 192
#define ln_seq_out_44 2
#define ffn_feature_out_45 192
#define ffn_seq_out_45 2
#define ffn_feature_out_45 192
#define ffn_seq_out_45 2
#define ffn_feature_out_45 192
#define ffn_seq_out_45 2
#define ln_feature_out_47 192
#define ln_seq_out_47 2
#define att_feature_out_48 192
#define att_seq_out_48 2
#define att_feature_out_48 192
#define att_seq_out_48 2
#define att_feature_out_48 192
#define att_seq_out_48 2
#define ln_feature_out_50 192
#define ln_seq_out_50 2
#define ffn_feature_out_51 192
#define ffn_seq_out_51 2
#define ffn_feature_out_51 192
#define ffn_seq_out_51 2
#define ffn_feature_out_51 192
#define ffn_seq_out_51 2
#define ln_feature_out_53 192
#define ln_seq_out_53 2
#define att_feature_out_54 192
#define att_seq_out_54 2
#define att_feature_out_54 192
#define att_seq_out_54 2
#define att_feature_out_54 192
#define att_seq_out_54 2
#define ln_feature_out_56 192
#define ln_seq_out_56 2
#define ffn_feature_out_57 192
#define ffn_seq_out_57 2
#define ffn_feature_out_57 192
#define ffn_seq_out_57 2
#define ffn_feature_out_57 192
#define ffn_seq_out_57 2
#define ln_feature_out_59 192
#define ln_seq_out_59 2
#define att_feature_out_60 192
#define att_seq_out_60 2
#define att_feature_out_60 192
#define att_seq_out_60 2
#define att_feature_out_60 192
#define att_seq_out_60 2
#define ln_feature_out_62 192
#define ln_seq_out_62 2
#define ffn_feature_out_63 192
#define ffn_seq_out_63 2
#define ffn_feature_out_63 192
#define ffn_seq_out_63 2
#define ffn_feature_out_63 192
#define ffn_seq_out_63 2
#define ln_feature_out_65 192
#define ln_seq_out_65 2
#define att_feature_out_66 192
#define att_seq_out_66 2
#define att_feature_out_66 192
#define att_seq_out_66 2
#define att_feature_out_66 192
#define att_seq_out_66 2
#define ln_feature_out_68 192
#define ln_seq_out_68 2
#define ffn_feature_out_69 192
#define ffn_seq_out_69 2
#define ffn_feature_out_69 192
#define ffn_seq_out_69 2
#define ffn_feature_out_69 192
#define ffn_seq_out_69 2
#define ln_feature_out_71 192
#define ln_seq_out_71 2
#define att_feature_out_72 192
#define att_seq_out_72 2
#define att_feature_out_72 192
#define att_seq_out_72 2
#define att_feature_out_72 192
#define att_seq_out_72 2
#define ln_feature_out_74 192
#define ln_seq_out_74 2
#define ffn_feature_out_75 192
#define ffn_seq_out_75 2
#define ffn_feature_out_75 192
#define ffn_seq_out_75 2
#define ffn_feature_out_75 192
#define ffn_seq_out_75 2
#define ln_feature_out_77 192
#define ln_seq_out_77 2
#define att_feature_out_78 192
#define att_seq_out_78 2
#define att_feature_out_78 192
#define att_seq_out_78 2
#define att_feature_out_78 192
#define att_seq_out_78 2
#define ln_feature_out_80 192
#define ln_seq_out_80 2
#define ffn_feature_out_81 192
#define ffn_seq_out_81 2
#define ffn_feature_out_81 192
#define ffn_seq_out_81 2
#define ffn_feature_out_81 192
#define ffn_seq_out_81 2
#define ln_feature_out_83 192
#define ln_seq_out_83 2
#define att_feature_out_84 192
#define att_seq_out_84 2
#define att_feature_out_84 192
#define att_seq_out_84 2
#define att_feature_out_84 192
#define att_seq_out_84 2
#define ln_feature_out_86 192
#define ln_seq_out_86 2
#define ffn_feature_out_87 192
#define ffn_seq_out_87 2
#define ffn_feature_out_87 192
#define ffn_seq_out_87 2
#define ln_feature_out_4 192
#define ln_seq_out_4 2

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_SAT,0>, 1*1> input_t;
typedef ap_fixed<80,32> layers_0_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_bias_t;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer17_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_0_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_0_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_0_norm1_mean_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_0_norm1_var_table_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<80,32> layers_0_self_attn_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask18_t;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer18_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_0_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_0_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_0_self_attn_scale_t;
typedef ap_fixed<24,4,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_row_sum_t;
typedef ap_fixed<24,4,AP_RND_CONV,AP_WRAP,0> layers_0_self_attn_out_proj_in_t;
typedef ap_uint<1> layer18_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer19_t;
typedef ap_fixed<80,32> layers_0_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer20_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_0_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_0_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_0_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_0_norm2_var_table_t;
typedef ap_uint<1> layer20_index;
typedef ap_fixed<80,32> layers_0_ffn_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_out_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer21_t;
typedef ap_fixed<24,4,AP_RND_CONV,AP_WRAP,0> layers_0_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_0_ffn_cdf_table_t;
typedef ap_uint<1> layer21_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer22_t;
typedef ap_fixed<80,32> layers_1_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer23_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_1_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_1_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_1_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_1_norm1_var_table_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<80,32> layers_1_self_attn_accum_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask24_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer24_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_1_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_1_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_1_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_1_self_attn_out_proj_in_t;
typedef ap_uint<1> layer24_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer25_t;
typedef ap_fixed<80,32> layers_1_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer26_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_1_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_1_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_1_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_1_norm2_var_table_t;
typedef ap_uint<1> layer26_index;
typedef ap_fixed<80,32> layers_1_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer27_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_1_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_1_ffn_cdf_table_t;
typedef ap_uint<1> layer27_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer28_t;
typedef ap_fixed<80,32> layers_2_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer29_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_2_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_2_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_2_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_2_norm1_var_table_t;
typedef ap_uint<1> layer29_index;
typedef ap_fixed<80,32> layers_2_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_bias_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask30_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer30_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_2_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_2_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_2_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_2_self_attn_out_proj_in_t;
typedef ap_uint<1> layer30_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer31_t;
typedef ap_fixed<80,32> layers_2_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer32_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_2_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_2_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_2_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_2_norm2_var_table_t;
typedef ap_uint<1> layer32_index;
typedef ap_fixed<80,32> layers_2_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer33_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_2_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_2_ffn_cdf_table_t;
typedef ap_uint<1> layer33_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer34_t;
typedef ap_fixed<80,32> layers_3_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer35_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_3_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_3_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_3_norm1_mean_t;
typedef ap_ufixed<18,2,AP_RND_CONV,AP_SAT,0> layers_3_norm1_var_table_t;
typedef ap_uint<1> layer35_index;
typedef ap_fixed<80,32> layers_3_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask36_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer36_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_3_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_3_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_3_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_3_self_attn_out_proj_in_t;
typedef ap_uint<1> layer36_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer37_t;
typedef ap_fixed<80,32> layers_3_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer38_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_3_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_3_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_3_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_3_norm2_var_table_t;
typedef ap_uint<1> layer38_index;
typedef ap_fixed<80,32> layers_3_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_out_proj_weight_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer39_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_3_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_3_ffn_cdf_table_t;
typedef ap_uint<1> layer39_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer40_t;
typedef ap_fixed<80,32> layers_4_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_4_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_4_norm1_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer41_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_4_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_4_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_4_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_4_norm1_var_table_t;
typedef ap_uint<1> layer41_index;
typedef ap_fixed<80,32> layers_4_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_in_proj_bias_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask42_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer42_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_4_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_4_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_4_self_attn_scale_t;
typedef ap_fixed<24,4,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_4_self_attn_out_proj_in_t;
typedef ap_uint<1> layer42_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer43_t;
typedef ap_fixed<80,32> layers_4_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_4_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_4_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer44_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_4_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_4_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_4_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_4_norm2_var_table_t;
typedef ap_uint<1> layer44_index;
typedef ap_fixed<80,32> layers_4_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_4_ffn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_4_ffn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_4_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_4_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer45_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_4_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_4_ffn_cdf_table_t;
typedef ap_uint<1> layer45_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer46_t;
typedef ap_fixed<80,32> layers_5_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_5_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_5_norm1_bias_t;
typedef nnet::array<ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0>, 1*1> layer47_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_5_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_5_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_5_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_5_norm1_var_table_t;
typedef ap_uint<1> layer47_index;
typedef ap_fixed<80,32> layers_5_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_in_proj_bias_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask48_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer48_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_5_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_5_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_5_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_5_self_attn_out_proj_in_t;
typedef ap_uint<1> layer48_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer49_t;
typedef ap_fixed<80,32> layers_5_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_5_norm2_scale_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_5_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer50_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_5_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_5_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_5_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_5_norm2_var_table_t;
typedef ap_uint<1> layer50_index;
typedef ap_fixed<80,32> layers_5_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_5_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_5_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_5_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_5_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer51_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_5_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_5_ffn_cdf_table_t;
typedef ap_uint<1> layer51_index;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer52_t;
typedef ap_fixed<80,32> layers_6_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_6_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_6_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer53_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_6_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_6_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_6_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_6_norm1_var_table_t;
typedef ap_uint<1> layer53_index;
typedef ap_fixed<80,32> layers_6_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask54_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer54_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_6_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_6_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_6_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_6_self_attn_out_proj_in_t;
typedef ap_uint<1> layer54_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer55_t;
typedef ap_fixed<80,32> layers_6_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_6_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_6_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer56_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_6_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_6_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_6_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_6_norm2_var_table_t;
typedef ap_uint<1> layer56_index;
typedef ap_fixed<80,32> layers_6_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_6_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_6_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_6_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_6_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer57_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_6_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_6_ffn_cdf_table_t;
typedef ap_uint<1> layer57_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer58_t;
typedef ap_fixed<80,32> layers_7_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_7_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_7_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer59_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_7_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_7_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_7_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_7_norm1_var_table_t;
typedef ap_uint<1> layer59_index;
typedef ap_fixed<80,32> layers_7_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_in_proj_bias_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask60_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer60_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_7_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_7_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_7_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_row_sum_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_7_self_attn_out_proj_in_t;
typedef ap_uint<1> layer60_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer61_t;
typedef ap_fixed<80,32> layers_7_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_7_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_7_norm2_bias_t;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer62_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_7_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_7_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_7_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_7_norm2_var_table_t;
typedef ap_uint<1> layer62_index;
typedef ap_fixed<80,32> layers_7_ffn_accum_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_7_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_7_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_7_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_7_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0>, 1*1> layer63_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_7_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_7_ffn_cdf_table_t;
typedef ap_uint<1> layer63_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer64_t;
typedef ap_fixed<80,32> layers_8_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_8_norm1_scale_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_8_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer65_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_8_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_8_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_8_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_8_norm1_var_table_t;
typedef ap_uint<1> layer65_index;
typedef ap_fixed<80,32> layers_8_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_in_proj_bias_t;
typedef ap_fixed<18,-1,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_out_proj_weight_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask66_t;
typedef nnet::array<ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0>, 1*1> layer66_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_8_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_8_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_8_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_8_self_attn_out_proj_in_t;
typedef ap_uint<1> layer66_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer67_t;
typedef ap_fixed<80,32> layers_8_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_8_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_8_norm2_bias_t;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer68_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_8_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_8_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_8_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_8_norm2_var_table_t;
typedef ap_uint<1> layer68_index;
typedef ap_fixed<80,32> layers_8_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_8_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_8_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_8_ffn_out_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_8_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer69_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_8_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_8_ffn_cdf_table_t;
typedef ap_uint<1> layer69_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer70_t;
typedef ap_fixed<80,32> layers_9_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_9_norm1_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_9_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer71_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_9_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_9_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_9_norm1_mean_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_9_norm1_var_table_t;
typedef ap_uint<1> layer71_index;
typedef ap_fixed<80,32> layers_9_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask72_t;
typedef nnet::array<ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0>, 1*1> layer72_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_9_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_9_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_9_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_row_sum_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_9_self_attn_out_proj_in_t;
typedef ap_uint<1> layer72_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer73_t;
typedef ap_fixed<80,32> layers_9_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_9_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_9_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer74_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_9_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_9_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_9_norm2_mean_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_9_norm2_var_table_t;
typedef ap_uint<1> layer74_index;
typedef ap_fixed<80,32> layers_9_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_9_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_9_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_9_ffn_out_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_9_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer75_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_9_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_9_ffn_cdf_table_t;
typedef ap_uint<1> layer75_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer76_t;
typedef ap_fixed<80,32> layers_10_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_10_norm1_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_10_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer77_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_10_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_10_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_10_norm1_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_10_norm1_var_table_t;
typedef ap_uint<1> layer77_index;
typedef ap_fixed<80,32> layers_10_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_in_proj_bias_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_out_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask78_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer78_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_10_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_10_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_10_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_row_sum_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_10_self_attn_out_proj_in_t;
typedef ap_uint<1> layer78_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer79_t;
typedef ap_fixed<80,32> layers_10_norm2_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_10_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_10_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer80_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_10_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_10_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_10_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_10_norm2_var_table_t;
typedef ap_uint<1> layer80_index;
typedef ap_fixed<80,32> layers_10_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_10_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_10_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_10_ffn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_10_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer81_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_10_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_10_ffn_cdf_table_t;
typedef ap_uint<1> layer81_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer82_t;
typedef ap_fixed<80,32> layers_11_norm1_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_norm1_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_norm1_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer83_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_11_norm1_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_11_norm1_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_11_norm1_mean_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_11_norm1_var_table_t;
typedef ap_uint<1> layer83_index;
typedef ap_fixed<80,32> layers_11_self_attn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_in_proj_weight_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_out_proj_weight_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_out_proj_bias_t;
typedef ap_uint<1> mask84_t;
typedef nnet::array<ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0>, 1*1> layer84_t;
typedef ap_ufixed<18,8,AP_RND_CONV,AP_SAT,0> layers_11_self_attn_exp_table_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_11_self_attn_inv_table_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_11_self_attn_scale_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_in_proj_out_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_row_sum_t;
typedef ap_fixed<24,3,AP_RND_CONV,AP_WRAP,0> layers_11_self_attn_out_proj_in_t;
typedef ap_uint<1> layer84_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer85_t;
typedef ap_fixed<80,32> layers_11_norm2_accum_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_norm2_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_norm2_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> layer86_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_11_norm2_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> layers_11_norm2_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> layers_11_norm2_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> layers_11_norm2_var_table_t;
typedef ap_uint<1> layer86_index;
typedef ap_fixed<80,32> layers_11_ffn_accum_t;
typedef ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0> layers_11_ffn_in_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_ffn_in_proj_bias_t;
typedef ap_fixed<18,1,AP_RND_CONV,AP_WRAP,0> layers_11_ffn_out_proj_weight_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> layers_11_ffn_out_proj_bias_t;
typedef nnet::array<ap_fixed<18,0,AP_RND_CONV,AP_WRAP,0>, 1*1> layer87_t;
typedef ap_fixed<24,2,AP_RND_CONV,AP_WRAP,0> layers_11_ffn_hidden_t;
typedef ap_ufixed<18,0,AP_RND_CONV,AP_SAT,0> layers_11_ffn_cdf_table_t;
typedef ap_uint<1> layer87_index;
typedef nnet::array<ap_fixed<18,5,AP_RND_CONV,AP_WRAP,0>, 1*1> layer88_t;
typedef ap_fixed<80,32> norm_accum_t;
typedef ap_fixed<18,3,AP_RND_CONV,AP_WRAP,0> norm_scale_t;
typedef ap_fixed<18,2,AP_RND_CONV,AP_WRAP,0> norm_bias_t;
typedef nnet::array<ap_fixed<18,4,AP_RND_CONV,AP_WRAP,0>, 1*1> result_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> norm_sum_t;
typedef ap_fixed<18,5,AP_RND_CONV,AP_SAT,0> norm_sum_sqr_t;
typedef ap_fixed<18,-3,AP_RND_CONV,AP_WRAP,0> norm_mean_t;
typedef ap_ufixed<18,1,AP_RND_CONV,AP_SAT,0> norm_var_table_t;
typedef ap_uint<1> layer4_index;

#endif
