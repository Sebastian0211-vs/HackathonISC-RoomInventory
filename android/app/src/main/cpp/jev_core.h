// JEV decision model (Tev1, a Qwen3.5 fine-tune) on llama.cpp: one forward pass over a rendered
// prompt, then the next-token logits of the option letters. Shared by the Android JNI wrapper and
// the desktop test in tools/jev_native_test.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

// Thrown by option_logits when JevModel::set_abort(true) stopped it.
struct JevAborted : std::runtime_error {
    JevAborted() : std::runtime_error("aborted") {}
};

class JevModel {
public:
    // Loads ggml's backend libraries from [dir] (Android: the app's native library folder, where each
    // CPU variant is its own library). Call once, before the first JevModel. No-op for static builds.
    static void load_backends(const std::string & dir);

    // Loads a GGUF file. Throws std::runtime_error on failure.
    JevModel(const std::string & path, int n_threads, int n_ctx);
    ~JevModel();
    JevModel(const JevModel &) = delete;
    JevModel & operator=(const JevModel &) = delete;

    // Logits of the next token after [prompt] (already in the model's chat format, special tokens
    // like <|im_start|> are parsed), for each single-token string in [options], in that order.
    // Throws std::runtime_error when the prompt is too long or an option is not one token, JevAborted
    // when set_abort(true) is called meanwhile (from any thread; the pass stops within milliseconds).
    std::vector<float> option_logits(const std::string & prompt, const std::vector<std::string> & options);

    // While true, every forward pass stops at once (for every model) and option_logits throws JevAborted.
    static void set_abort(bool abort);

    int last_prompt_tokens() const { return last_tokens_; }
    std::string describe() const;

private:
    std::vector<int> tokenize(const std::string & text, bool parse_special) const;

    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int n_ctx_ = 0;
    int last_tokens_ = 0;
};
