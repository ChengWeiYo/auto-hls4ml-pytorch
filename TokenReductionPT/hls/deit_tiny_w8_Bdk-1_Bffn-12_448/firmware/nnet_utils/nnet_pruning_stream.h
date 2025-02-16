#ifndef NNET_PRUNING_H_
#define NNET_PRUNING_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include "hls_streamofblocks.h"
#include <math.h>
#include <cmath>
#include <iostream>

namespace nnet {

template<class T, class idx_T, const idx_T num_tokens>
void sort_topk(const T attentions[], idx_T topk_idx[], int k) {
    #pragma HLS INLINE
    bool selected[num_tokens] = {0};
    for (int i = 0; i < k; i++) {
        // #pragma HLS UNROLL
        T max_val = attentions[0];
        int max_idx = 0;
        for (int j = 1; j < num_tokens; j++) {
            if (!selected[j] && attentions[j][0] > max_val[0]) {
                max_val = attentions[j];
                max_idx = j;
            }
        }
        selected[max_idx] = true;
        topk_idx[i] = max_idx;
    }
}

template<class data_T, class idx_T, class config_T>
void PruningLayer(
    hls::stream<data_T> &data_in,
    // hls::stream<data_T> &attn,
    hls::stream<data_T> &data_out,
    const float keep_rate
) {
    
    const unsigned int N = CONFIG_T::seq_len;
    const unsigned int D = CONFIG_T::embed_dim;
    // const unsigned int H = CONFIG_T::head_dim/tf_H;
    data_T token_buffer[config_T::max_tokens][config_T::embed_dim];
    data_T row_buffer[config_T::max_tokens];
    data_T data;
    data_T attn_token_buffer[config_T::max_tokens][config_T::embed_dim];
    data_T attn_row_buffer[config_T::max_tokens];
    data_T attn_data;
    data_T out_data;
    const idx_T keep_tokens = int(config_T::batch_size * keep_rate);
    idx_T topk_idx[config_T::max_tokens];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            // #pragma HLS PIPELINE II=1
    // for (int buffer_count = 0; buffer_count < config_T::max_tokens / data_T::size ; buffer_count++) {
            // std::cout << "buffer_count: " << buffer_count << std::endl;
            data = data_in.read();
            // attn_data = attn.read();
            if (i == 0) // cls_token
                data_out.write(data);
            else {
                for (int k = 0; k < data_T::size; k++) {
                    #pragma HLS UNROLL
                    row_buffer[j][k] = data[k];
                    // attn_row_buffer[j][k] = attn_data[k];
                }
                if (j == D-1) {
                    for (int ii = 0; ii < D; ii++) {
                        #pragma HLS UNROLL
                        for (int jj = 0; jj < data_T::size; jj++) {
                            #pragma HLS UNROLL
                            token_buffer[i][ii][jj] = row_buffer[ii][jj];
                            // attn_token_buffer[i][ii][jj] = attn_row_buffer[ii][jj];
                        }
                    }
                    if (i % config_T::batch_size == 0) {
                        // sort_topk<data_T, idx_T, config_T::batch_size>(token_buffer, attn_token_buffer, topk_idx, keep_tokens);
                        // output top-k tokens
                        for (int k = 0; k < keep_tokens; k++) {
                            for (int ii = 0; ii < D; ii++) {
                                for (int jj = 0; jj < data_T::size; jj++) {
                                    #pragma HLS UNROLL
                                    out_data[jj] = token_buffer[k][ii][jj];
                                }
                                data_out.write(out_data);
                            }
                        }
                    }
                }
            }
        }
    }     
}
}

#endif
