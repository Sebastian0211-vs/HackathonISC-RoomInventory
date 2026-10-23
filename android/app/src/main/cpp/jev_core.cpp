#include "jev_core.h"

#include "ggml-backend.h"
#include "llama.h"

#include <atomic>
#include <mutex>
#include <stdexcept>

namespace {
std::once_flag backend_once;
std::atomic<bool> abort_flag{false};

// ggml checks this between graph nodes, so a pause stops the CPU work almost at once.
bool should_abort(void *) {
    return abort_flag.load(std::memory_order_relaxed);
}
}  // namespace

void JevModel::set_abort(bool abort) {
    abort_flag.store(abort);
}

void JevModel::load_backends(const std::string & dir) {
    ggml_backend_load_all_from_path(dir.c_str());
}

JevModel::JevModel(const std::string & path, int n_threads, int n_ctx) : n_ctx_(n_ctx) {
    std::call_once(backend_once, [] {
        llama_backend_init();
    });
    if (ggml_backend_dev_count() == 0) throw std::runtime_error("no ggml backend loaded");

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU, weights memory-mapped (the default load mode)
    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) throw std::runtime_error("cannot load " + path);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_ctx;  // the whole prompt goes in one llama_decode call
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf = true;
    cp.abort_callback = should_abort;
    cp.abort_callback_data = nullptr;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("cannot create a context for " + path);
    }
    vocab_ = llama_model_get_vocab(model_);
}

JevModel::~JevModel() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::vector<int> JevModel::tokenize(const std::string & text, bool parse_special) const {
    int n = -llama_tokenize(vocab_, text.data(), (int32_t) text.size(), nullptr, 0, false, parse_special);
    if (n <= 0) return {};
    std::vector<int> out((size_t) n);
    int got = llama_tokenize(vocab_, text.data(), (int32_t) text.size(), out.data(), n, false, parse_special);
    if (got < 0) throw std::runtime_error("tokenization failed");
    out.resize((size_t) got);
    return out;
}

std::vector<float> JevModel::option_logits(const std::string & prompt, const std::vector<std::string> & options) {
    std::vector<int> option_ids;
    option_ids.reserve(options.size());
    for (const auto & o : options) {
        auto ids = tokenize(o, false);
        if (ids.size() != 1) throw std::runtime_error("option \"" + o + "\" is not a single token");
        option_ids.push_back(ids[0]);
    }

    std::vector<int> tokens = tokenize(prompt, true);
    last_tokens_ = (int) tokens.size();
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    if ((int) tokens.size() >= n_ctx_) {
        throw std::runtime_error("prompt has " + std::to_string(tokens.size()) + " tokens, context is " +
                                 std::to_string(n_ctx_));
    }

    if (abort_flag.load()) throw JevAborted();
    // Every call starts from an empty state (Qwen3.5's recurrent layers cannot be rolled back partially).
    llama_memory_clear(llama_get_memory(ctx_), true);

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == tokens.size();  // only the last position is needed
    }
    batch.n_tokens = (int32_t) tokens.size();
    int rc = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (rc == 2 || (rc != 0 && abort_flag.load())) throw JevAborted();
    if (rc != 0) throw std::runtime_error("llama_decode failed (" + std::to_string(rc) + ")");

    const float * logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) throw std::runtime_error("no logits");
    std::vector<float> out;
    out.reserve(option_ids.size());
    for (int id : option_ids) out.push_back(logits[id]);
    return out;
}

std::string JevModel::describe() const {
    char desc[256];
    llama_model_desc(model_, desc, sizeof(desc));
    std::string cpu;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) cpu = ggml_backend_dev_description(dev);
    }
    return std::string(desc) + ", " + std::to_string(llama_model_n_params(model_) / 1000000) + "M params" +
           (cpu.empty() ? "" : ", CPU " + cpu);
}
