#pragma once

#include "tokenizer.hpp"
#include "cnpy.h"

namespace tokenizer
{
    std::pair<std::vector<std::vector<float>>, std::vector<int>> get_hailo_input(const std::vector<std::string>& input_text) {
        Tokenizer tokenizer;

        std::vector<std::vector<int>> tokens = tokenizer.tokenize(input_text);

        cnpy::NpyArray arr = cnpy::npy_load("tokenizer/ViT-L-14_laion2b_s32b_b82k.npy");
        float* loaded_data = arr.data<float>();
        int vocab_size = 49408;
        int embedding_dim = 768;

        std::vector<float> embedding_weights(loaded_data, loaded_data + vocab_size * embedding_dim);

        int num_tokens = (int)tokens[0].size();

        std::vector<std::vector<float>> hailo_input(tokens.size(), std::vector<float>(num_tokens * embedding_dim));
        std::vector<int> last_tokens(tokens.size(), 0);

        // Lookup embeddings for each token in each set
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].size() != num_tokens) {
                throw std::invalid_argument("Each tokenized text must contain exactly 77 tokens.");
            }

            for (int j = 0; j < num_tokens; ++j) {
                int token_id = tokens[i][j];

                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::out_of_range("Token ID out of vocabulary range");
                }

                if (token_id > 0) {
                    last_tokens[i]++;
                }

                for (int k = 0; k < embedding_dim; ++k) {
                    hailo_input[i][j * embedding_dim + k] = embedding_weights[token_id * embedding_dim + k];
                }
            }
        }

        return std::make_pair(hailo_input, last_tokens);
    }
} // namespace tokenizer