
#include "tokenizer.hpp"


std::string Tokenizer::basic_clean(const std::string& text) {
    return text.empty() ? text : text.substr(0, text.find_last_not_of(" \n\r\t") + 1);
}

std::string Tokenizer::whitespace_clean(const std::string& text) {
    std::string cleaned = std::regex_replace(text, std::regex("\\s+"), " ");
    cleaned = cleaned.empty() ? cleaned : cleaned.substr(0, cleaned.find_last_not_of(" \n\r\t") + 1);
    return cleaned;
}

std::string Tokenizer::canonicalize_text(const std::string& text) {
    std::string result = text;
    result.erase(std::remove_if(result.begin(), result.end(), ::ispunct), result.end());
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string Tokenizer::_clean_canonicalize(const std::string& x) {
    return canonicalize_text(basic_clean(x));
}

std::string Tokenizer::_clean_lower(const std::string& x) {
    return whitespace_clean(basic_clean(x));
}

std::string Tokenizer::_clean_whitespace(const std::string& x) {
    return whitespace_clean(basic_clean(x));
}

std::function<std::string(const std::string&)> Tokenizer::get_clean_fn(const std::string& type) {
    if (type == "canonicalize") {
        return _clean_canonicalize;
    } else if (type == "lower") {
        return _clean_lower;
    } else if (type == "whitespace") {
        return _clean_whitespace;
    } else {
        throw std::invalid_argument("Invalid clean function (" + type + ").");
    }
}


std::string Tokenizer::read_text_file(const std::string &filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();  // Load the entire file into the buffer
    file.close();
    return buffer.str();
}

std::vector<std::string> Tokenizer::split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


std::vector<std::pair<std::string, std::string>> Tokenizer::get_pairs(const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;

    if (word.size() < 2) {
        return pairs;
    }

    std::string prev_char = word[0];

    for (size_t i = 1; i < word.size(); ++i) {
        const std::string& char_curr = word[i];
        pairs.emplace_back(prev_char, char_curr);
        prev_char = char_curr;
    }

    return pairs;
}

std::string char_to_string(char c) {
    return std::string(1, c);
}

std::string Tokenizer::bpe(const std::string& token) {

    if (token.empty()) {
        throw std::invalid_argument("Token must not be empty");
    }

    if (this->cache.find(token) != this->cache.end()) {
        return cache[token];
    }

    std::vector<std::string> word;

    std::string word_part = token.substr(0, token.size() - 1); // All but the last character
    std::string last_part = char_to_string(token[token.size() - 1]) + "</w>";  // Last character + '</w>'

    for (char c : word_part) {
        word.push_back(char_to_string(c));
    }
    word.push_back(last_part);

    auto pairs = get_pairs(word);

    if (pairs.empty()) {
        return token + "</w>";
    }

    while (true) {
        // Find the lowest ranked bigram
        auto bigram = *std::min_element(pairs.begin(), pairs.end(), [this](const auto &a, const auto &b) {
            int rank_a = this->bpe_ranks.count(a) ? this->bpe_ranks[a] : std::numeric_limits<int>::max();
            int rank_b = this->bpe_ranks.count(b) ? this->bpe_ranks[b] : std::numeric_limits<int>::max();
            return rank_a < rank_b;
        });

        if (this->bpe_ranks.find(bigram) == this->bpe_ranks.end()) {
            break;
        }

        std::string first = bigram.first;
        std::string second = bigram.second;

        // Merge pairs into new_word
        std::vector<std::string> new_word;
        new_word.reserve(word.size());

        size_t i = 0;
        while (i < word.size()) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            new_word.insert(new_word.end(), word.begin() + i, it);
            i = std::distance(word.begin(), it);

            if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                ++i;
            }
        }

        word = std::move(new_word);

        if (word.size() == 1) {
            break;
        } else {
            pairs = get_pairs(word);
        }
    }

    // Join the word and add it to the cache
    std::string result = std::accumulate(word.begin(), word.end(), std::string(), [](const std::string& a, const std::string& b) {
        return a.empty() ? b : a + " " + b;
    });

    cache[token] = result;
    return result;
}


Tokenizer::Tokenizer(){

    std::string content = read_text_file("tokenizer/bpe_simple_vocab_16e6.txt");

    std::vector<std::string> merges = split(content, '\n');

    // Select the desired range [1:49152-256-2+1] (from index 1 to 48895 inclusive)
    int start_idx = 1;
    int end_idx = 49152 - 256 - 2 + 1;

    merges.erase(merges.begin() + end_idx, merges.end());
    merges.erase(merges.begin(), merges.begin() + start_idx);

    std::vector<std::string> vocab;
    std::vector<std::pair<std::string, std::string>> merge_pairs;

    for (const auto& p : BYTES_ENCODER) {
        vocab.push_back(p.second);
    }


    for (const auto& p : BYTES_ENCODER) {
        vocab.push_back(p.second + "</w>");
    }

    for (const auto& merge : merges) {
        auto merge_pair = split(merge, ' ');
        vocab.push_back(merge_pair[0] + merge_pair[1]);
        merge_pairs.push_back({merge_pair[0], merge_pair[1]});
    }

    std::vector<std::string> special_tokens = {"<start_of_text>", "<end_of_text>"};

    vocab.push_back(special_tokens[0]);
    vocab.push_back(special_tokens[1]);

    std::map<std::string, int> encoder;

    for (int i = 0; i < vocab.size(); i++) {
        encoder[vocab[i]] = i;
    }

    this->encoder = encoder;

    std::map<int, std::string> decoder;

    for (int i = 0; i < vocab.size(); i++) {
        decoder[i] = vocab[i];
    }

    this->decoder = decoder;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    for (int i = 0; i < merge_pairs.size(); i++) {
        bpe_ranks[merge_pairs[i]] = i;
    }

    this->bpe_ranks = bpe_ranks;

    std::map<std::string, std::string> cache = {{special_tokens[0], special_tokens[0]}, {special_tokens[1], special_tokens[1]}};

    this->cache = cache;

    std::string special = special_tokens[0] + "|" + special_tokens[1]; 

    std::regex pat(
        special + R"('s|'t|'re|'ve|'m|'ll|'d|([A-Za-z]+)|([0-9])|([^\sA-Za-z0-9]+))",
        std::regex_constants::icase
    );

    this->pat = pat;

    this->vocabulary_size = encoder.size();
    
    this->all_special_ids = {encoder[special_tokens[0]], encoder[special_tokens[1]]};

    this->sot_token_id = all_special_ids[0];
    this->eot_token_id = all_special_ids[1];

    this->context_length = 77;

    this->clean_fn = get_clean_fn("lower");
}

Tokenizer::~Tokenizer(){
    // Destructor
}

size_t Tokenizer::get_vocab_size() {
    return this->vocabulary_size;
}

std::vector<int> Tokenizer::encode(std::string text){
    std::vector<int> bpe_tokens;
    auto cleaned_text = this->clean_fn(text);

    std::sregex_iterator iter(cleaned_text.begin(), cleaned_text.end(), this->pat);
    std::sregex_iterator end;

    while (iter != end) {
        std::string token = (*iter).str();

        std::string encoded_token;
        for (unsigned char b : token) {
            encoded_token += BYTES_ENCODER_MAP[b];
        }

        std::string bpe_encoded_token = bpe(encoded_token);
        std::istringstream bpe_token_stream(bpe_encoded_token);
        std::string bpe_token;

        while (std::getline(bpe_token_stream, bpe_token, ' ')) {
            bpe_tokens.push_back(encoder[bpe_token]);
        }

        ++iter;
    }

    return bpe_tokens;
}


std::vector<std::vector<int>> Tokenizer::tokenize(std::vector<std::string> text){
    
    std::vector<std::vector<int>> all_tokens;
    all_tokens.reserve(text.size()); // Reserve capacity to avoid frequent reallocations
    
    for (const auto& txt : text) {
        std::vector<int> tokens;
        tokens.reserve(this->context_length); // Reserve capacity to avoid frequent reallocations
        tokens.push_back(this->sot_token_id);
        auto encoded = encode(txt);
        tokens.insert(tokens.end(), encoded.begin(), encoded.end());
        tokens.push_back(eot_token_id);
        all_tokens.push_back(tokens);
    }

    std::vector<std::vector<int>> result(all_tokens.size(), std::vector<int>(this->context_length, 0));

    for (size_t i = 0; i < all_tokens.size(); ++i) {
        auto& tokens = all_tokens[i];
        if (tokens.size() > this->context_length) {
            tokens.resize(this->context_length);
            tokens.back() = this->eot_token_id;
        }
        std::copy(tokens.begin(), tokens.end(), result[i].begin());
    }

    return result;
}
