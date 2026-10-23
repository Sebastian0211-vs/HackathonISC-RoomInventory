// jev_test model.gguf prompts.json  ->  one JSON line per prompt: {"logits": [...], "tokens": n, "ms": t}
// prompts.json: [{"prompt": "...", "n": number of options}] as written by export_jev.py; letters A, B, C...
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "jev_core.h"

namespace {

void put_utf8(std::string & v, unsigned cp) {
    if (cp < 0x80) {
        v += (char) cp;
    } else if (cp < 0x800) {
        v += (char) (0xC0 | (cp >> 6));
        v += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        v += (char) (0xE0 | (cp >> 12));
        v += (char) (0x80 | ((cp >> 6) & 0x3F));
        v += (char) (0x80 | (cp & 0x3F));
    } else {
        v += (char) (0xF0 | (cp >> 18));
        v += (char) (0x80 | ((cp >> 12) & 0x3F));
        v += (char) (0x80 | ((cp >> 6) & 0x3F));
        v += (char) (0x80 | (cp & 0x3F));
    }
}

struct Reader {
    const std::string & s;
    size_t i = 0;
    void skip() { while (i < s.size() && std::isspace((unsigned char) s[i])) i++; }
    bool str(std::string & v) {
        skip();
        if (s[i] != '"') return false;
        i++;
        v.clear();
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') { v += c; continue; }
            char e = s[i++];
            if (e == 'n') v += '\n';
            else if (e == 't') v += '\t';
            else if (e == 'r') v += '\r';
            else if (e == 'b') v += '\b';
            else if (e == 'f') v += '\f';
            else if (e == 'u') {
                unsigned cp = std::stoul(s.substr(i, 4), nullptr, 16);
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && s[i] == '\\' && s[i + 1] == 'u') {
                    unsigned lo = std::stoul(s.substr(i + 2, 4), nullptr, 16);
                    i += 6;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                put_utf8(v, cp);
            } else v += e;
        }
        i++;
        return true;
    }
    int integer() {
        skip();
        size_t j = i;
        while (std::isdigit((unsigned char) s[i])) i++;
        return std::stoi(s.substr(j, i - j));
    }
};

bool parse(const std::string & s, std::vector<std::pair<std::string, int>> & out) {
    Reader r{s};
    r.skip();
    if (s[r.i++] != '[') return false;
    while (true) {
        r.skip();
        if (s[r.i] == ']') return true;
        if (s[r.i] == ',') { r.i++; continue; }
        if (s[r.i++] != '{') return false;
        std::string prompt;
        int n = 0;
        while (true) {
            r.skip();
            if (s[r.i] == '}') { r.i++; break; }
            if (s[r.i] == ',') { r.i++; continue; }
            std::string key;
            if (!r.str(key)) return false;
            r.skip();
            r.i++;  // ':'
            if (key == "prompt") {
                if (!r.str(prompt)) return false;
            } else {
                n = r.integer();
            }
        }
        out.emplace_back(prompt, n);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: jev_test model.gguf prompts.json\n";
        return 2;
    }
    std::ifstream f(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<std::pair<std::string, int>> prompts;
    if (!parse(ss.str(), prompts)) {
        std::cerr << "cannot parse " << argv[2] << "\n";
        return 2;
    }
    int threads = (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency() / 2));
    JevModel m(argv[1], threads, 2048);
    std::cerr << m.describe() << ", " << threads << " threads\n";
    const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWX";
    for (auto & [prompt, n] : prompts) {
        std::vector<std::string> opts;
        for (int k = 0; k < n; k++) opts.emplace_back(1, letters[k]);
        auto t0 = std::chrono::steady_clock::now();
        auto logits = m.option_logits(prompt, opts);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("{\"logits\": [");
        for (size_t k = 0; k < logits.size(); k++) std::printf("%s%.6f", k ? ", " : "", logits[k]);
        std::printf("], \"tokens\": %d, \"ms\": %.1f}\n", m.last_prompt_tokens(), ms);
        std::fflush(stdout);
    }
    return 0;
}
