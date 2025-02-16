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

// template<class T, class idx_T, const idx_T num_tokens>
// void sort_topk(const T attentions[], idx_T topk_idx[], int k) {
//     #pragma HLS INLINE
//     bool selected[num_tokens] = {0};
//     for (int i = 0; i < k; i++) {
//         // #pragma HLS UNROLL
//         T max_val = attentions[0];
//         int max_idx = 0;
//         for (int j = 1; j < num_tokens; j++) {
//             if (!selected[j] && attentions[j][0] > max_val[0]) {
//                 max_val = attentions[j];
//                 max_idx = j;
//             }
//         }
//         selected[max_idx] = true;
//         topk_idx[i] = max_idx;
//     }
// }

template<class data_T, class res_T, typename CONFIG_T>
void PruningLayer(
    hls::stream<data_T> &data_in,
    hls::stream<res_T> &data_out,
    const float keep_rate,
    const unsigned int N
) {
    const unsigned int D = CONFIG_T::embed_dim;
    const unsigned int keep_tokens = static_cast<unsigned int>(std::ceil((N-1) * keep_rate));

    struct token_buf_t {
        data_T data[CONFIG_T::embed_dim];
    };

    hls::stream<token_buf_t> token_buffer_stream;
    #pragma HLS STREAM variable=token_buffer_stream depth=1

    hls::stream<token_buf_t> sorted_token_buffer_stream;
    #pragma HLS STREAM variable=sorted_token_buffer_stream depth=1

    #pragma HLS DATAFLOW

    // Stage 1: Read input data and store in token buffer stream
    read_input:
    for (int i = 0; i < N; i++) {
        token_buf_t token_buffer;
        for (int j = 0; j < D; j++) {
            #pragma HLS PIPELINE II=1
            data_T data = data_in.read();
            for (int k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                token_buffer.data[j][k] = data[k];
            }
        }
        token_buffer_stream.write(token_buffer);
    }

    // Stage 2: Sort tokens (excluding cls_token)
    sort_tokens:
    for (int i = 0; i < N; i++) {
        token_buf_t token_buffer = token_buffer_stream.read();
        if (i == 0) {
            sorted_token_buffer_stream.write(token_buffer); // Pass cls_token directly
        } else {
            // Insert sorting logic here
            // For simplicity, assuming tokens are sorted based on the first element of data
            // You can replace this with your actual sorting logic
            static token_buf_t token_array[CONFIG_T::max_tokens];
            // #pragma HLS ARRAY_PARTITION variable=token_array complete dim=1
            token_array[i-1] = token_buffer;
            if (i == N-1) {
                // Sort the tokens
                // for (int m = 0; m < N-1; m++) {
                //     for (int n = m+1; n < N-1; n++) {
                //         if (token_array[m].data[0] < token_array[n].data[0]) {
                //             token_buf_t temp = token_array[m];
                //             token_array[m] = token_array[n];
                //             token_array[n] = temp;
                //         }
                //     }
                // }
                // Write sorted tokens to stream
                for (int k = 0; k < keep_tokens; k++) {
                    sorted_token_buffer_stream.write(token_array[k]);
                }
            }
        }
    }

    // Stage 3: Process sorted token buffer stream and write output data
    process_tokens:
    for (int i = 0; i < keep_tokens + 1; i++) { // +1 to include cls_token
        token_buf_t token_buffer = sorted_token_buffer_stream.read();
        for (int j = 0; j < D; j++) {
            #pragma HLS PIPELINE II=1
            res_T out_data;
            for (int k = 0; k < data_T::size; k++) {
                #pragma HLS UNROLL
                out_data[k] = token_buffer.data[j][k];
            }
            data_out.write(out_data);
        }
    }
}
}

#endif
