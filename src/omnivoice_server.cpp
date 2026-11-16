#include "omnivoice.h"

#include "ggml.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

void quiet_ggml_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::cerr << text;
}

// --- WAV encoding ---------------------------------------------------------

std::string encode_wav(const std::vector<float> & samples, int sample_rate) {
    const size_t n = samples.size();
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align = num_channels * bits_per_sample / 8;
    const uint32_t data_size = static_cast<uint32_t>(n * num_channels * bits_per_sample / 8);
    const uint32_t file_size = 36 + data_size;

    std::string buf;
    buf.reserve(44 + data_size);

    auto write_u8 = [&](uint8_t v) { buf.push_back(static_cast<char>(v)); };
    auto write_u16 = [&](uint16_t v) { write_u8(v & 0xFF); write_u8((v >> 8) & 0xFF); };
    auto write_u32 = [&](uint32_t v) { write_u8(v & 0xFF); write_u8((v >> 8) & 0xFF); write_u8((v >> 16) & 0xFF); write_u8((v >> 24) & 0xFF); };

    // RIFF header
    buf.append("RIFF", 4);
    write_u32(file_size);
    buf.append("WAVE", 4);

    // fmt chunk
    buf.append("fmt ", 4);
    write_u32(16); // chunk size
    write_u16(1);  // PCM format
    write_u16(num_channels);
    write_u32(static_cast<uint32_t>(sample_rate));
    write_u32(byte_rate);
    write_u16(block_align);
    write_u16(bits_per_sample);

    // data chunk
    buf.append("data", 4);
    write_u32(data_size);

    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        write_u8(static_cast<uint8_t>(pcm & 0xFF));
        write_u8(static_cast<uint8_t>((pcm >> 8) & 0xFF));
    }
    return buf;
}

// --- PCM raw output (24kHz 16-bit signed little-endian) -------------------

std::string encode_pcm(const std::vector<float> & samples) {
    std::string buf;
    buf.reserve(samples.size() * 2);
    for (float s : samples) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
        buf.push_back(static_cast<char>(pcm & 0xFF));
        buf.push_back(static_cast<char>((pcm >> 8) & 0xFF));
    }
    return buf;
}

// --- Helpers --------------------------------------------------------------

std::string format_content_type(const std::string & format) {
    if (format == "mp3")  return "audio/mpeg";
    if (format == "pcm")  return "audio/pcm";
    return "audio/wav";
}

std::string encode_audio(const std::vector<float> & samples, int sample_rate, const std::string & format) {
    if (format == "mp3") return omnivoice::encode_mp3_mono_f32(samples, sample_rate);
    if (format == "pcm") return encode_pcm(samples);
    return encode_wav(samples, sample_rate);
}

std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim_whitespace(const std::string & s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string json_error(const std::string & message, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", message}, {"type", type}}}}.dump();
}

void set_json(httplib::Response & res, const json & value, int status = 200) {
    res.status = status;
    res.set_content(value.dump(), "application/json");
}

bool has_field(const json & j, const std::string & key) {
    return j.contains(key) && !j.at(key).is_null();
}

std::string get_opt_string(const json & j, const std::string & key, const std::string & def = "") {
    if (!has_field(j, key) || !j.at(key).is_string()) return def;
    return j.at(key).get<std::string>();
}

std::optional<double> get_opt_number(const json & j, const std::string & key) {
    if (!has_field(j, key)) return std::nullopt;
    if (!j.at(key).is_number()) return std::nullopt;
    return j.at(key).get<double>();
}

std::optional<bool> get_opt_bool(const json & j, const std::string & key) {
    if (!has_field(j, key)) return std::nullopt;
    if (j.at(key).is_boolean()) return j.at(key).get<bool>();
    if (j.at(key).is_number()) return j.at(key).get<double>() != 0.0;
    return std::nullopt;
}

std::string lower_extension(const fs::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

bool is_reference_audio_path(const fs::path & path) {
    const std::string ext = lower_extension(path);
    return ext == ".wav" || ext == ".mp3" || ext == ".flac";
}

bool looks_like_preset_name(const std::string & name) {
    return !name.empty()
        && name.find('/') == std::string::npos
        && name.find('\\') == std::string::npos
        && name != "."
        && name != "..";
}

int parse_int_option(const std::string & value, const std::string & name) {
    size_t parsed = 0;
    int out = 0;
    try {
        out = std::stoi(value, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
    if (parsed != value.size()) throw std::runtime_error("invalid integer for " + name + ": " + value);
    return out;
}

// --- CapCut SSML & Voice Helpers ------------------------------------------

std::string unescape_xml(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '&') {
            if (s.compare(i, 4, "&lt;") == 0) { out.push_back('<'); i += 3; continue; }
            if (s.compare(i, 4, "&gt;") == 0) { out.push_back('>'); i += 3; continue; }
            if (s.compare(i, 5, "&amp;") == 0) { out.push_back('&'); i += 4; continue; }
            if (s.compare(i, 6, "&quot;") == 0) { out.push_back('"'); i += 5; continue; }
            if (s.compare(i, 6, "&apos;") == 0) { out.push_back('\''); i += 5; continue; }
        }
        out.push_back(s[i]);
    }
    return out;
}

std::string strip_xml_tags(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool in_tag = false;
    for (char c : s) {
        if (c == '<') { in_tag = true; continue; }
        if (c == '>') { in_tag = false; continue; }
        if (!in_tag) out.push_back(c);
    }
    return out;
}

std::string extract_xml_attr(const std::string & tag, const std::string & attr_name) {
    size_t pos = 0;
    while ((pos = tag.find(attr_name, pos)) != std::string::npos) {
        const bool boundary = (pos == 0) || std::isspace(static_cast<unsigned char>(tag[pos - 1]));
        size_t after = pos + attr_name.size();
        if (boundary && after < tag.size() && tag[after] == '=') {
            ++after;
            if (after < tag.size() && (tag[after] == '"' || tag[after] == '\'')) {
                const char quote = tag[after++];
                const size_t end = tag.find(quote, after);
                if (end != std::string::npos) {
                    return unescape_xml(tag.substr(after, end - after));
                }
            }
        }
        pos = after;
    }
    return "";
}

struct CapCutSegment {
    std::string text;
    std::string voice_type;
    std::string resource_id;
    float rate = 1.0f;
};

float clamp_speed(float speed) {
    if (speed < 0.25f) return 0.25f;
    if (speed > 4.0f) return 4.0f;
    return speed;
}

std::vector<CapCutSegment> parse_capcut_ssml(const std::string & ssml) {
    std::vector<CapCutSegment> segments;
    size_t pos = 0;
    while (true) {
        const size_t open_start = ssml.find("<voice", pos);
        if (open_start == std::string::npos) break;
        const size_t after_kw = open_start + 6;
        if (after_kw < ssml.size() && !std::isspace(static_cast<unsigned char>(ssml[after_kw])) && ssml[after_kw] != '>') {
            pos = after_kw;
            continue;
        }
        const size_t open_end = ssml.find('>', after_kw);
        if (open_end == std::string::npos) break;
        const std::string open_tag = ssml.substr(open_start, open_end - open_start + 1);
        const size_t close_start = ssml.find("</voice>", open_end + 1);
        const std::string inner = (close_start == std::string::npos)
            ? ssml.substr(open_end + 1)
            : ssml.substr(open_end + 1, close_start - (open_end + 1));

        CapCutSegment seg;
        seg.voice_type = extract_xml_attr(open_tag, "name");
        seg.resource_id = extract_xml_attr(open_tag, "resource_id");

        std::string text_source = inner;
        const size_t pros_start = inner.find("<prosody");
        if (pros_start != std::string::npos) {
            const size_t pros_end = inner.find('>', pros_start);
            if (pros_end != std::string::npos) {
                const std::string pros_tag = inner.substr(pros_start, pros_end - pros_start + 1);
                const std::string rate_str = extract_xml_attr(pros_tag, "rate");
                if (!rate_str.empty()) {
                    try { seg.rate = clamp_speed(std::stof(rate_str)); } catch (const std::exception &) {}
                }
                const size_t pros_close = inner.find("</prosody>", pros_end + 1);
                text_source = (pros_close == std::string::npos)
                    ? inner.substr(pros_end + 1)
                    : inner.substr(pros_end + 1, pros_close - (pros_end + 1));
            }
        }
        seg.text = trim_whitespace(unescape_xml(strip_xml_tags(text_source)));
        if (!seg.text.empty()) segments.push_back(std::move(seg));

        pos = (close_start == std::string::npos) ? ssml.size() : close_start + 8;
    }

    if (segments.empty()) {
        std::string plain = trim_whitespace(unescape_xml(strip_xml_tags(ssml)));
        if (!plain.empty()) {
            CapCutSegment seg;
            seg.text = std::move(plain);
            segments.push_back(std::move(seg));
        }
    }
    return segments;
}

void infer_from_capcut_voice(const std::string & voice_type, std::string & out_lang, std::string & out_instruct) {
    const std::string v = to_lower_ascii(voice_type);
    out_lang.clear();
    out_instruct.clear();

    if (v.find("vivn") != std::string::npos || v.rfind("vi_", 0) == 0) out_lang = "vi";
    else if (v.find("enus") != std::string::npos || v.find("engb") != std::string::npos
             || v.find("enau") != std::string::npos || v.rfind("en_", 0) == 0) out_lang = "en";
    else if (v.find("zhcn") != std::string::npos || v.find("zhtw") != std::string::npos
             || v.rfind("zh_", 0) == 0) out_lang = "zh";
    else if (v.find("jajp") != std::string::npos || v.rfind("ja_", 0) == 0
             || v.rfind("jp_", 0) == 0) out_lang = "ja";
    else if (v.find("kokr") != std::string::npos || v.rfind("ko_", 0) == 0) out_lang = "ko";
    else if (v.find("eses") != std::string::npos || v.find("esmx") != std::string::npos
             || v.rfind("es_", 0) == 0) out_lang = "es";
    else if (v.find("frfr") != std::string::npos || v.rfind("fr_", 0) == 0) out_lang = "fr";
    else if (v.find("dede") != std::string::npos || v.rfind("de_", 0) == 0) out_lang = "de";
    else if (v.find("thth") != std::string::npos || v.rfind("th_", 0) == 0) out_lang = "th";
    else if (v.find("idid") != std::string::npos || v.rfind("id_", 0) == 0) out_lang = "id";
    else if (v.find("ptbr") != std::string::npos || v.rfind("pt_", 0) == 0) out_lang = "pt";
    else if (v.find("ruru") != std::string::npos || v.rfind("ru_", 0) == 0) out_lang = "ru";

    if (v.find("female") != std::string::npos || v.find("huong") != std::string::npos) {
        out_instruct = "female";
    } else if (v.find("male") != std::string::npos) {
        out_instruct = "male";
    }
    if (v.find("child") != std::string::npos || v.find("_dsp") != std::string::npos) {
        if (!out_instruct.empty()) out_instruct += ", ";
        out_instruct += "child";
    }
}

bool looks_like_omnivoice_instruct(const std::string & s) {
    const std::string v = to_lower_ascii(s);
    for (const char * kw : {"male", "female", "child", "teenager", "young adult", "middle-aged",
                            "elderly", "pitch", "whisper", "accent"}) {
        if (v.find(kw) != std::string::npos) return true;
    }
    return s.find("话") != std::string::npos;
}

std::string generate_random_hex(size_t num_bytes = 16) {
    static std::atomic<uint64_t> counter{0};
    static thread_local std::mt19937_64 rng([]() {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd())
            ^ static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    }());
    const uint64_t seq = ++counter;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    size_t written = 0;
    while (written < num_bytes) {
        uint64_t val = rng() ^ (seq * 0x9e3779b97f4a7c15ULL);
        for (int b = 0; b < 8 && written < num_bytes; ++b, ++written) {
            oss << std::setw(2) << ((val >> (b * 8)) & 0xFF);
        }
    }
    return oss.str();
}

// --- CapCut Async Task Queue ----------------------------------------------

struct CapCutArtifact {
    std::string bytes;
    std::string format;
    std::string text;
    int duration_ms = 0;
};

struct CapCutTask {
    std::string id;
    std::string token;
    std::string bind_id;
    std::string status = "running"; // "running", "succeed", "failed"
    std::string error;
    std::vector<CapCutArtifact> artifacts;
};

class WorkerPool {
public:
    WorkerPool() : worker_([this]() { run(); }) {}
    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool & operator=(const WorkerPool &) = delete;

    void enqueue(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(fn));
        }
        cv_.notify_one();
    }

private:
    void run() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop();
            }
            if (job) job();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> queue_;
    bool stop_ = false;
    std::thread worker_;
};

// --- Voice preset system --------------------------------------------------
// Maps voice names to ref_audio/ref_text files on disk.
// Users can place WAV/MP3/FLAC + txt pairs in a voices/ directory.

struct VoicePreset {
    std::string ref_audio_path;
    std::string ref_text;
};

std::string voices_dir;

VoicePreset load_voice_preset(const std::string & name) {
    VoicePreset p;
    if (voices_dir.empty() || !looks_like_preset_name(name)) return p;
    const fs::path base = fs::path(voices_dir) / name;

    std::ifstream tf(base.string() + ".txt");
    if (tf) std::getline(tf, p.ref_text);

    for (const char * ext : {".wav", ".mp3", ".flac"}) {
        const fs::path audio_path = base.string() + ext;
        if (fs::is_regular_file(audio_path)) {
            p.ref_audio_path = audio_path.string();
            break;
        }
    }
    return p;
}

json list_voice_presets() {
    json out = {{"object", "list"}, {"data", json::array()}};
    if (voices_dir.empty()) return out;

    const fs::path dir(voices_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return out;

    for (const auto & entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file() || !is_reference_audio_path(entry.path())) continue;

        const std::string name = entry.path().stem().string();
        VoicePreset preset = load_voice_preset(name);
        out["data"].push_back({
            {"id", name},
            {"object", "voice"},
            {"has_ref_text", !preset.ref_text.empty()},
            {"ref_text", preset.ref_text},
        });
    }
    return out;
}

// --- Shared Synthesis Parameter Helpers -----------------------------------

void apply_generation_options(const json & j, omnivoice::SynthesisParams & params) {
    if (auto v = get_opt_number(j, "num_step"))
        params.generation.num_step = std::max(1, static_cast<int>(*v));
    if (auto v = get_opt_number(j, "guidance_scale"))
        params.generation.guidance_scale = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "t_shift"))
        params.generation.t_shift = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "layer_penalty_factor"))
        params.generation.layer_penalty_factor = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "position_temperature"))
        params.generation.position_temperature = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "class_temperature"))
        params.generation.class_temperature = static_cast<float>(*v);
    if (auto v = get_opt_bool(j, "denoise"))
        params.generation.denoise = *v;
    if (auto v = get_opt_bool(j, "preprocess_prompt"))
        params.generation.preprocess_prompt = *v;
    if (auto v = get_opt_bool(j, "postprocess_output"))
        params.generation.postprocess_output = *v;
    if (auto v = get_opt_number(j, "duration"))
        params.duration = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "audio_chunk_duration"))
        params.generation.audio_chunk_duration = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "audio_chunk_threshold"))
        params.generation.audio_chunk_threshold = static_cast<float>(*v);
    if (auto v = get_opt_number(j, "seed"))
        params.generation.seed = static_cast<uint64_t>(*v);
}

bool apply_voice_and_ref_options(const json & j, omnivoice::SynthesisParams & params, std::string & err_out) {
    if (has_field(j, "language")) {
        params.language = get_opt_string(j, "language");
    }
    if (auto auto_voice = get_opt_bool(j, "auto_voice")) {
        params.auto_voice = *auto_voice;
    }

    const std::string voice = get_opt_string(j, "voice");
    std::string instructions = get_opt_string(j, "instructions");
    if (instructions.empty()) instructions = get_opt_string(j, "instruct");

    bool voice_is_instruct = false;
    if (!voice.empty()) {
        VoicePreset preset = load_voice_preset(voice);
        if (!preset.ref_audio_path.empty()) {
            params.ref_audio_path = preset.ref_audio_path;
            params.ref_text = preset.ref_text;
        } else {
            params.instruct = voice;
            voice_is_instruct = true;
        }
    }

    if (has_field(j, "ref_audio")) {
        const std::string ref = get_opt_string(j, "ref_audio");
        if (!ref.empty()) {
            if (fs::is_regular_file(ref)) {
                if (!is_reference_audio_path(ref)) {
                    err_out = "'ref_audio' must point to a WAV, MP3, or FLAC file";
                    return false;
                }
                params.ref_audio_path = ref;
            } else {
                VoicePreset p = load_voice_preset(ref);
                if (!p.ref_audio_path.empty()) {
                    params.ref_audio_path = p.ref_audio_path;
                    if (params.ref_text.empty()) params.ref_text = p.ref_text;
                } else {
                    err_out = "'ref_audio' is neither a readable WAV path nor a configured voice preset";
                    return false;
                }
            }
        }
    }
    if (has_field(j, "ref_text")) {
        params.ref_text = get_opt_string(j, "ref_text");
    }

    if (!params.ref_audio_path.empty() && params.ref_text.empty()) {
        err_out = "'ref_text' is required when using reference audio";
        return false;
    }

    if (!instructions.empty() && voice_is_instruct) {
        params.instruct = voice + ", " + instructions;
    } else if (!instructions.empty()) {
        params.instruct = instructions;
    }
    return true;
}

// --- OpenAPI / Swagger UI -------------------------------------------------

const char * kSwaggerUiHtml = R"ov_html(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OmniVoice &amp; CapCut TTS API — Swagger UI</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body { margin: 0; background: #fafafa; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function () {
      SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>)ov_html";

std::string load_openapi_spec() {
    for (const char * candidate : {"docs/openapi.json", "../docs/openapi.json"}) {
        std::ifstream ifs(candidate, std::ios::binary);
        if (ifs) {
            std::ostringstream oss;
            oss << ifs.rdbuf();
            const std::string content = oss.str();
            if (!content.empty()) return content;
        }
    }
    static const char * kOpenApiPart1 = R"ov_spec({
  "openapi": "3.0.3",
  "info": {
    "title": "OmniVoice & CapCut Compatible TTS API",
    "version": "1.0.0",
    "description": "Standalone C++ text-to-speech HTTP server exposing both an OpenAI-compatible TTS API (/v1/audio/speech) and CapCut common-task compatible TTS endpoints (/lv/v1/common_task/new, /lv/v1/common_task/query, /lv/v1/tts_artifact/{task_id}/{filename})."
  },
  "servers": [
    {
      "url": "/",
      "description": "Current OmniVoice Server (same origin)"
    },
    {
      "url": "http://localhost:8000",
      "description": "Local Docker OmniVoice Server"
    },
    {
      "url": "https://editor-api-sg.capcutapi.com",
      "description": "CapCut Cloud Backend (requires signed device headers)"
    }
  ],
  "tags": [
    {
      "name": "OpenAI-Compatible TTS",
      "description": "Synchronous speech synthesis and voice preset inspection"
    },
    {
      "name": "CapCut Common-Task TTS",
      "description": "Asynchronous task-based TTS endpoints compatible with capcut-tts-api"
    },
    {
      "name": "System",
      "description": "Health and service status"
    }
  ],
  "paths": {
    "/health": {
      "get": {
        "tags": ["System"],
        "summary": "Health check",
        "operationId": "getHealth",
        "responses": {
          "200": {
            "description": "Server is ready",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HealthResponse" },
                "example": { "status": "ok" }
              }
            }
          }
        }
      }
    },
    "/v1/models": {
      "get": {
        "tags": ["OpenAI-Compatible TTS"],
        "summary": "List loaded models",
        "operationId": "listModels",
        "responses": {
          "200": {
            "description": "Available model list",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/ModelListResponse" }
              }
            }
          }
        }
      }
    },
    "/v1/audio/voices": {
      "get": {
        "tags": ["OpenAI-Compatible TTS"],
        "summary": "List voice cloning presets from --voices-dir",
        "operationId": "listVoicePresets",
        "responses": {
          "200": {
            "description": "Configured voice presets",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/VoicePresetListResponse" }
              }
            }
          }
        }
      }
    },
    "/v1/audio/speech": {
      "post": {
        "tags": ["OpenAI-Compatible TTS"],
        "summary": "Synthesize speech audio (OpenAI-compatible)",
        "operationId": "createSpeech",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/SpeechRequest" },
              "examples": {
                "english_voice_design": {
                  "summary": "English voice design (MP3)",
                  "value": {
                    "input": "Hello! Welcome to OmniVoice running in Docker.",
                    "voice": "female, young adult",
                    "response_format": "mp3",
                    "num_step": 8,
                    "speed": 1.0
                  }
                },
                "vietnamese_tts": {
                  "summary": "Vietnamese synthesis (WAV)",
                  "value": {
                    "input": "Xin chào! Đây là giọng nói tiếng Việt từ OmniVoice.",
                    "voice": "female, young adult",
                    "language": "vi",
                    "response_format": "wav",
                    "num_step": 8
                  }
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Synthesized binary audio stream",
            "content": {
              "audio/mpeg": {
                "schema": { "type": "string", "format": "binary" }
              },
              "audio/wav": {
                "schema": { "type": "string", "format": "binary" }
              },
              "audio/pcm": {
                "schema": { "type": "string", "format": "binary" }
              }
            }
          },
          "400": {
            "description": "Invalid request parameters",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/OpenAIError" }
              }
            }
          },
          "500": {
            "description": "Synthesis failure",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/OpenAIError" }
              }
            }
          }
        }
      }
    },
    "/lv/v1/common_task/new": {
      "post": {
        "tags": ["CapCut Common-Task TTS"],
        "summary": "Submit an asynchronous CapCut-compatible TTS task",
        "description": "Accepts CapCut common-task JSON with SSML (<speak><voice><prosody>) and optional payload.omnivoice overrides. Returns a task id and token to poll via /lv/v1/common_task/query. STT tasks (req_key=cc_audio_subtitle_asr) return ret='4001'.",
        "operationId": "createCapCutCommonTask",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/CapCutCommonTaskNewRequest" },
              "examples": {
                "capcut_ssml_inline": {
                  "summary": "CapCut SSML with inline JSON payload",
                  "value": {
                    "bind_id": "11111111-2222-3333-4444-555555555555",
                    "can_queue": true,
                    "enter_from": "text_to_speech",
                    "tasks": [
                      {
                        "context": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "req_key": "sami_text_to_speech",
                        "task_version": "v3",
                        "payload": {
                          "audio_format": "mp3",
                          "scene": "text_to_speech",
                          "ssml": "<speak version=\"1.0\" xml:lang=\"en-US\"><voice name=\"BV421_vivn_streaming\" resource_id=\"7252594014782755330\"><prosody rate=\"1.0\">Xin chào từ OmniVoice!</prosody></voice></speak>",
                          "omnivoice": {
                            "language": "vi",
                            "instructions": "female, young adult",
                            "num_step": 8
                          }
                        }
                      }
                    ]
                  }
                },
                "capcut_ssml_string": {
                  "summary": "CapCut SDK format (stringified payload)",
                  "value": {
                    "bind_id": "11111111-2222-3333-4444-555555555555",
                    "can_queue": true,
                    "enter_from": "text_to_speech",
                    "tasks": [
                      {
                        "context": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        "req_key": "sami_text_to_speech",
                        "task_version": "v3",
                        "payload": "{\"audio_format\":\"mp3\",\"scene\":\"text_to_speech\",\"ssml\":\"<speak version=\\\"1.0\\\" xml:lang=\\\"en-US\\\"><voice name=\\\"BV074_streaming\\\" resource_id=\\\"7102355709945188865\\\"><prosody rate=\\\"1.0\\\">Hello from CapCut compatible endpoint.</prosody></voice></speak>\"}"
                      }
                    ]
                  }
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "CapCut task envelope (ret='0' on success, non-zero ret on application error)",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/CapCutCommonTaskResponse" },
                "example": {
                  "ret": "0",
                  "errmsg": "",
                  "data": {
                    "bind_id": "11111111-2222-3333-4444-555555555555",
                    "tasks": [
                      {
                        "id": "26eab49bc26ba8107ebd57ab1ce6fbb1",
                        "token": "e978f94dea08285e0770d621ec599a8c",
                        "bind_id": "11111111-2222-3333-4444-555555555555",
                        "req_key": "sami_text_to_speech",
                        "task_version": "v3",
                        "status": "running"
                      }
                    ]
                  }
                }
              }
            }
          },
          "400": {
            "description": "Malformed JSON body",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/OpenAIError" }
              }
            }
          }
        }
      }
    },)ov_spec";
    static const char * kOpenApiPart2 = R"ov_spec(
    "/lv/v1/common_task/query": {
      "post": {
        "tags": ["CapCut Common-Task TTS"],
        "summary": "Poll a CapCut-compatible TTS task",
        "description": "Queries the status of a task created via /lv/v1/common_task/new. When status is 'succeed', payload is a JSON string containing audio_subtitles[] with downloadable artifact URLs.",
        "operationId": "queryCapCutCommonTask",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/CapCutCommonTaskQueryRequest" },
              "example": {
                "tasks": [
                  {
                    "id": "26eab49bc26ba8107ebd57ab1ce6fbb1",
                    "token": "e978f94dea08285e0770d621ec599a8c",
                    "bind_id": "11111111-2222-3333-4444-555555555555",
                    "req_key": "sami_text_to_speech",
                    "task_version": "v3"
                  }
                ]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Task status and completed artifact URLs",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/CapCutCommonTaskResponse" },
                "example": {
                  "ret": "0",
                  "errmsg": "",
                  "data": {
                    "bind_id": "11111111-2222-3333-4444-555555555555",
                    "tasks": [
                      {
                        "id": "26eab49bc26ba8107ebd57ab1ce6fbb1",
                        "token": "e978f94dea08285e0770d621ec599a8c",
                        "bind_id": "11111111-2222-3333-4444-555555555555",
                        "req_key": "sami_text_to_speech",
                        "task_version": "v3",
                        "status": "succeed",
                        "payload": "{\"audio_subtitles\":[{\"url\":\"http://localhost:8000/lv/v1/tts_artifact/26eab49bc26ba8107ebd57ab1ce6fbb1/0.mp3\",\"speech_url\":\"http://localhost:8000/lv/v1/tts_artifact/26eab49bc26ba8107ebd57ab1ce6fbb1/0.mp3\",\"audio_url\":\"http://localhost:8000/lv/v1/tts_artifact/26eab49bc26ba8107ebd57ab1ce6fbb1/0.mp3\",\"text\":\"Hello!\",\"duration_ms\":1480}]}"
                      }
                    ]
                  }
                }
              }
            }
          }
        }
      }
    },
    "/lv/v1/tts_artifact/{task_id}/{filename}": {
      "get": {
        "tags": ["CapCut Common-Task TTS"],
        "summary": "Download a synthesized segment artifact",
        "operationId": "downloadCapCutArtifact",
        "parameters": [
          {
            "name": "task_id",
            "in": "path",
            "required": true,
            "schema": { "type": "string" },
            "description": "Task ID returned by /lv/v1/common_task/new"
          },
          {
            "name": "filename",
            "in": "path",
            "required": true,
            "schema": { "type": "string", "example": "0.mp3" },
            "description": "Zero-based segment filename (e.g. 0.mp3 or 0.wav)"
          }
        ],
        "responses": {
          "200": {
            "description": "Synthesized audio file",
            "content": {
              "audio/mpeg": {
                "schema": { "type": "string", "format": "binary" }
              },
              "audio/wav": {
                "schema": { "type": "string", "format": "binary" }
              },
              "audio/pcm": {
                "schema": { "type": "string", "format": "binary" }
              }
            }
          },
          "404": {
            "description": "Artifact or task not found",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/OpenAIError" }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "HealthResponse": {
        "type": "object",
        "properties": {
          "status": { "type": "string", "example": "ok" }
        }
      },
      "ModelListResponse": {
        "type": "object",
        "properties": {
          "object": { "type": "string", "example": "list" },
          "data": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": { "type": "string", "example": "omnivoice" },
                "object": { "type": "string", "example": "model" },
                "owned_by": { "type": "string", "example": "local" }
              }
            }
          }
        }
      },
      "VoicePresetListResponse": {
        "type": "object",
        "properties": {
          "object": { "type": "string", "example": "list" },
          "data": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": { "type": "string" },
                "object": { "type": "string", "example": "voice" },
                "has_ref_text": { "type": "boolean" },
                "ref_text": { "type": "string" }
              }
            }
          }
        }
      },
      "SpeechRequest": {
        "type": "object",
        "required": ["input"],
        "properties": {
          "input": { "type": "string", "description": "Text to synthesize" },
          "voice": { "type": "string", "description": "Preset name from --voices-dir or voice design attributes (e.g. 'female, young adult')" },
          "instructions": { "type": "string", "description": "Voice design attributes" },
          "instruct": { "type": "string", "description": "Alias for instructions" },
          "language": { "type": "string", "description": "Language name or ISO code (e.g. 'en', 'vi', 'zh', 'ja')" },
          "response_format": { "type": "string", "enum": ["wav", "mp3", "pcm"], "default": "wav" },
          "speed": { "type": "number", "minimum": 0.25, "maximum": 4.0, "default": 1.0 },
          "ref_audio": { "type": "string", "description": "Reference WAV/MP3/FLAC path or preset name for voice cloning" },
          "ref_text": { "type": "string", "description": "Transcript of ref_audio (required when ref_audio is set)" },
          "auto_voice": { "type": "boolean", "default": false },
          "num_step": { "type": "integer", "minimum": 1, "description": "Unmasking steps (higher = better quality, slower)" },
          "guidance_scale": { "type": "number", "default": 2.0 },
          "t_shift": { "type": "number", "default": 0.1 },
          "layer_penalty_factor": { "type": "number", "default": 5.0 },
          "position_temperature": { "type": "number", "default": 5.0 },
          "class_temperature": { "type": "number", "default": 0.0 },
          "duration": { "type": "number", "description": "Fixed output duration in seconds" },
          "denoise": { "type": "boolean", "default": true },
          "preprocess_prompt": { "type": "boolean", "default": true },
          "postprocess_output": { "type": "boolean", "default": true },
          "audio_chunk_duration": { "type": "number", "default": 15.0 },
          "audio_chunk_threshold": { "type": "number", "default": 30.0 },
          "seed": { "type": "integer" }
        }
      },
      "CapCutCommonTaskNewRequest": {
        "type": "object",
        "required": ["tasks"],
        "properties": {
          "bind_id": { "type": "string" },
          "can_queue": { "type": "boolean", "default": true },
          "enter_from": { "type": "string", "default": "text_to_speech" },
          "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "required": ["payload"],
              "properties": {
                "context": { "type": "string" },
                "req_key": { "type": "string", "default": "sami_text_to_speech" },
                "task_version": { "type": "string", "default": "v3" },
                "payload": {
                  "description": "JSON object or JSON-serialized string containing ssml, audio_format, and optional omnivoice overrides",
                  "oneOf": [
                    { "type": "string" },
                    { "type": "object" }
                  ]
                }
              }
            }
          }
        }
      },
      "CapCutCommonTaskQueryRequest": {
        "type": "object",
        "required": ["tasks"],
        "properties": {
          "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object",
              "required": ["id", "token"],
              "properties": {
                "id": { "type": "string" },
                "token": { "type": "string" },
                "bind_id": { "type": "string" },
                "req_key": { "type": "string", "default": "sami_text_to_speech" },
                "task_version": { "type": "string", "default": "v3" }
              }
            }
          }
        }
      },
      "CapCutCommonTaskResponse": {
        "type": "object",
        "properties": {
          "ret": { "type": "string", "example": "0" },
          "errmsg": { "type": "string", "example": "" },
          "data": {
            "type": "object",
            "properties": {
              "bind_id": { "type": "string" },
              "tasks": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "id": { "type": "string" },
                    "token": { "type": "string" },
                    "bind_id": { "type": "string" },
                    "req_key": { "type": "string" },
                    "task_version": { "type": "string" },
                    "status": { "type": "string", "enum": ["running", "succeed", "failed"] },
                    "payload": { "type": "string" },
                    "errmsg": { "type": "string" }
                  }
                }
              }
            }
          }
        }
      },
      "OpenAIError": {
        "type": "object",
        "properties": {
          "error": {
            "type": "object",
            "properties": {
              "message": { "type": "string" },
              "type": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
)ov_spec";
    return std::string(kOpenApiPart1) + kOpenApiPart2;
}

// --- Server ---------------------------------------------------------------

struct ServerConfig {
    std::string model_path;
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string backend = "cpu";
    int device = 0;
    int threads = 4;
    int default_num_step = 32;
};

void usage() {
    std::cerr
        << "usage: omnivoice-server --model MODEL.gguf [options]\n"
        << "options:\n"
        << "  --host HOST         listen host (default: 0.0.0.0)\n"
        << "  --port PORT         listen port (default: 8000)\n"
        << "  --voices-dir DIR    directory with voice presets (name.wav/mp3/flac + name.txt)\n"
        << "  --device cpu|cuda[:N]\n"
        << "  --backend cpu|cuda\n"
        << "  --threads N         number of CPU threads (default: 4)\n"
        << "  --num-step N        default generation steps when omitted (default: 32)\n";
}

} // namespace

int main(int argc, char ** argv) {
    ggml_log_set(quiet_ggml_log, nullptr);

    try {
    ServerConfig cfg;
    if (const char * env_step = std::getenv("OMNIVOICE_DEFAULT_NUM_STEP")) {
        if (*env_step) cfg.default_num_step = std::max(1, parse_int_option(env_step, "OMNIVOICE_DEFAULT_NUM_STEP"));
    } else if (const char * env_step2 = std::getenv("OMNIVOICE_NUM_STEP")) {
        if (*env_step2) cfg.default_num_step = std::max(1, parse_int_option(env_step2, "OMNIVOICE_NUM_STEP"));
    }

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const char * name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--model") cfg.model_path = need("--model");
        else if (key == "--host") cfg.host = need("--host");
        else if (key == "--port") cfg.port = parse_int_option(need("--port"), "--port");
        else if (key == "--voices-dir") voices_dir = need("--voices-dir");
        else if (key == "--backend") {
            cfg.backend = need("--backend");
            if (cfg.backend == "rocm" || cfg.backend == "hip") cfg.backend = "cuda";
        } else if (key == "--device") {
            std::string dev = need("--device");
            if (dev == "cpu") cfg.backend = "cpu";
            else if (dev.rfind("cuda", 0) == 0) {
                cfg.backend = "cuda";
                auto pos = dev.find(':');
                if (pos != std::string::npos) cfg.device = parse_int_option(dev.substr(pos + 1), "--device");
            } else {
                throw std::runtime_error("unsupported --device: " + dev);
            }
        } else if (key == "--threads" || key == "-t") cfg.threads = parse_int_option(need("--threads"), "--threads");
        else if (key == "--num-step" || key == "--default-num-step") {
            cfg.default_num_step = std::max(1, parse_int_option(need(key.c_str()), key));
        }
        else if (key == "--help" || key == "-h") { usage(); return 0; }
        else { std::cerr << "unknown argument: " << key << "\n"; usage(); return 2; }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "error: --model is required\n";
        usage();
        return 2;
    }

    // Load model once — stays in VRAM/RAM for the lifetime of the server
    std::cerr << "[init] loading model from " << cfg.model_path << " ...\n";
    omnivoice::RuntimeOptions options;
    options.backend = cfg.backend;
    options.device = cfg.device;
    options.threads = cfg.threads;

    omnivoice::OmniVoiceRuntime runtime(cfg.model_path, options);
    std::cerr << "[init] model loaded (backend=" << cfg.backend
              << " sample_rate=" << runtime.sample_rate() << ")\n";

    // Mutex to serialize generate() calls (GPU inference is not thread-safe)
    std::mutex gen_mutex;

    // Async CapCut-compatible common_task queue
    constexpr size_t kMaxCapCutTasks = 100;
    std::mutex tasks_mutex;
    std::unordered_map<std::string, std::shared_ptr<CapCutTask>> capcut_tasks;
    std::deque<std::string> capcut_task_order;
    WorkerPool capcut_worker;

    // --- HTTP server ---
    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        set_json(res, {{"status", "ok"}});
    });

    // OpenAPI 3.0.3 spec and Swagger UI
    const std::string openapi_spec = load_openapi_spec();
    svr.Get("/openapi.json", [openapi_spec](const httplib::Request &, httplib::Response & res) {
        res.status = 200;
        res.set_content(openapi_spec, "application/json; charset=utf-8");
    });

    auto serve_swagger = [](const httplib::Request &, httplib::Response & res) {
        res.status = 200;
        res.set_content(kSwaggerUiHtml, "text/html; charset=utf-8");
    };
    svr.Get("/docs", serve_swagger);
    svr.Get("/swagger", serve_swagger);

    // GET /v1/models — list available models
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response & res) {
        set_json(res, {
            {"object", "list"},
            {"data", json::array({
                {{"id", "omnivoice"}, {"object", "model"}, {"owned_by", "local"}},
            })},
        });
    });

    // GET /v1/audio/voices — list available voice presets
    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        try {
            set_json(res, list_voice_presets());
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json_error(std::string("failed to list voices: ") + e.what()), "application/json");
        }
    });

    // POST /v1/audio/speech — the main endpoint
    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        // Parse JSON body
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(json_error("request body is empty"), "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error & e) {
            res.status = 400;
            res.set_content(json_error(std::string("invalid JSON: ") + e.what()), "application/json");
            return;
        }
        if (!body.is_object()) {
            res.status = 400;
            res.set_content(json_error("request body must be a JSON object"), "application/json");
            return;
        }

        // --- Required parameters ---
        std::string input_text = get_opt_string(body, "input");
        if (input_text.empty()) {
            res.status = 400;
            res.set_content(json_error("'input' is required and must be a non-empty string"), "application/json");
            return;
        }

        std::string voice = get_opt_string(body, "voice");

        // --- Response format ---
        std::string response_format = get_opt_string(body, "response_format", "wav");
        if (response_format != "wav" && response_format != "pcm" && response_format != "mp3") {
            res.status = 400;
            res.set_content(json_error("unsupported response_format '" + response_format + "'; supported formats are 'wav', 'pcm', and 'mp3'"), "application/json");
            return;
        }

        // --- Speed ---
        float speed = clamp_speed(static_cast<float>(get_opt_number(body, "speed").value_or(1.0)));

        // --- Build SynthesisParams ---
        omnivoice::SynthesisParams params;
        params.text = input_text;
        params.speed = speed;
        params.generation.num_step = cfg.default_num_step;

        std::string voice_err;
        if (!apply_voice_and_ref_options(body, params, voice_err)) {
            res.status = 400;
            res.set_content(json_error(voice_err), "application/json");
            return;
        }
        apply_generation_options(body, params);

        // --- Generate ---
        std::cerr << "[request] input=\"" << input_text.substr(0, 80)
                  << (input_text.size() > 80 ? "..." : "")
                  << "\" voice=\"" << voice
                  << "\" format=" << response_format
                  << " speed=" << speed << "\n";

        omnivoice::Audio audio;
        const auto gen_start = std::chrono::steady_clock::now();
        try {
            std::lock_guard<std::mutex> lock(gen_mutex);
            audio = runtime.generate(params);
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json_error(std::string("generation failed: ") + e.what()), "application/json");
            std::cerr << "[error] " << e.what() << "\n";
            return;
        }
        const double gen_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_start).count();

        // --- Encode response ---
        std::string audio_bytes = encode_audio(audio.samples, audio.sample_rate, response_format);

        const double audio_seconds = audio.sample_rate > 0
            ? double(audio.samples.size()) / double(audio.sample_rate)
            : 0.0;
        const double rtf = audio_seconds > 0.0 ? gen_seconds / audio_seconds : 0.0;
        std::cerr << std::fixed << std::setprecision(3)
                  << "[response] gen=" << gen_seconds << "s"
                  << " audio=" << audio_seconds << "s"
                  << " rtf=" << rtf
                  << " size=" << audio_bytes.size() << " (" << response_format << ")\n";

        res.status = 200;
        res.set_content(audio_bytes, format_content_type(response_format));
    });

    // POST /lv/v1/common_task/new — CapCut common-task compatible TTS submission
    svr.Post("/lv/v1/common_task/new", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(json_error("request body is empty"), "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error & e) {
            res.status = 400;
            res.set_content(json_error(std::string("invalid JSON: ") + e.what()), "application/json");
            return;
        }
        if (!body.is_object()) {
            res.status = 400;
            res.set_content(json_error("request body must be a JSON object"), "application/json");
            return;
        }

        const std::string enter_from = get_opt_string(body, "enter_from");
        if (!body.contains("tasks") || !body.at("tasks").is_array() || body.at("tasks").empty()
            || !body.at("tasks").at(0).is_object()) {
            set_json(res, {{"ret", "4000"}, {"errmsg", "missing tasks[0] in request body"}, {"data", json::object()}});
            return;
        }

        const json & task_obj = body.at("tasks").at(0);
        const std::string req_key = get_opt_string(task_obj, "req_key");
        if (req_key == "cc_audio_subtitle_asr" || enter_from == "asr") {
            set_json(res, {
                {"ret", "4001"},
                {"errmsg", "STT (cc_audio_subtitle_asr) is not supported by omnivoice-server (TTS only). Use the CapCut backend for STT."},
                {"data", json::object()},
            });
            return;
        }

        json payload = json::object();
        if (task_obj.contains("payload")) {
            const auto & raw_payload = task_obj.at("payload");
            if (raw_payload.is_string()) {
                try {
                    payload = json::parse(raw_payload.get<std::string>());
                } catch (const json::parse_error & e) {
                    set_json(res, {{"ret", "4000"}, {"errmsg", std::string("invalid task payload JSON: ") + e.what()}, {"data", json::object()}});
                    return;
                }
            } else if (raw_payload.is_object()) {
                payload = raw_payload;
            }
        }
        if (!payload.is_object()) {
            set_json(res, {{"ret", "4000"}, {"errmsg", "task payload must be a JSON object"}, {"data", json::object()}});
            return;
        }

        std::string audio_format = to_lower_ascii(get_opt_string(payload, "audio_format", "mp3"));
        if (audio_format != "mp3" && audio_format != "wav" && audio_format != "pcm") {
            audio_format = "mp3";
        }

        const json ov = (payload.contains("omnivoice") && payload.at("omnivoice").is_object())
            ? payload.at("omnivoice")
            : json::object();
        if (has_field(ov, "response_format")) {
            const std::string ov_fmt = to_lower_ascii(get_opt_string(ov, "response_format"));
            if (ov_fmt == "mp3" || ov_fmt == "wav" || ov_fmt == "pcm") {
                audio_format = ov_fmt;
            }
        }

        std::vector<CapCutSegment> segments;
        const std::string ssml = get_opt_string(payload, "ssml");
        if (!ssml.empty()) {
            segments = parse_capcut_ssml(ssml);
        }
        if (segments.empty() && ov.contains("texts") && ov.at("texts").is_array()) {
            for (const auto & item : ov.at("texts")) {
                if (item.is_string()) {
                    std::string t = trim_whitespace(item.get<std::string>());
                    if (!t.empty()) {
                        CapCutSegment seg;
                        seg.text = std::move(t);
                        segments.push_back(std::move(seg));
                    }
                }
            }
        }
        if (segments.empty() && has_field(ov, "input")) {
            std::string t = trim_whitespace(get_opt_string(ov, "input"));
            if (!t.empty()) {
                CapCutSegment seg;
                seg.text = std::move(t);
                segments.push_back(std::move(seg));
            }
        }
        if (segments.empty() && has_field(payload, "text")) {
            std::string t = trim_whitespace(get_opt_string(payload, "text"));
            if (!t.empty()) {
                CapCutSegment seg;
                seg.text = std::move(t);
                segments.push_back(std::move(seg));
            }
        }
        if (segments.empty()) {
            set_json(res, {{"ret", "4000"}, {"errmsg", "no non-empty TTS text found in SSML or payload"}, {"data", json::object()}});
            return;
        }

        // Validate omnivoice override fields up front so invalid ref_audio fails immediately
        omnivoice::SynthesisParams override_probe;
        std::string voice_err;
        if (!ov.empty() && !apply_voice_and_ref_options(ov, override_probe, voice_err)) {
            set_json(res, {{"ret", "4000"}, {"errmsg", voice_err}, {"data", json::object()}});
            return;
        }

        std::string bind_id = get_opt_string(body, "bind_id");
        if (bind_id.empty()) bind_id = generate_random_hex(16);

        auto task = std::make_shared<CapCutTask>();
        task->id = generate_random_hex(16);
        task->token = generate_random_hex(16);
        task->bind_id = bind_id;
        task->status = "running";

        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            capcut_tasks[task->id] = task;
            capcut_task_order.push_back(task->id);
            while (capcut_task_order.size() > kMaxCapCutTasks) {
                const std::string oldest_id = capcut_task_order.front();
                auto it = capcut_tasks.find(oldest_id);
                if (it != capcut_tasks.end() && it->second->status == "running") break;
                capcut_tasks.erase(oldest_id);
                capcut_task_order.pop_front();
            }
        }

        std::cerr << "[capcut-task] queued id=" << task->id
                  << " segments=" << segments.size()
                  << " format=" << audio_format << "\n";

        capcut_worker.enqueue([&, task, segments = std::move(segments), ov, audio_format]() {
            try {
                std::vector<CapCutArtifact> produced;
                produced.reserve(segments.size());

                for (size_t idx = 0; idx < segments.size(); ++idx) {
                    const CapCutSegment & seg = segments[idx];
                    omnivoice::SynthesisParams params;
                    params.text = seg.text;
                    params.speed = clamp_speed(seg.rate);
                    params.generation.num_step = cfg.default_num_step;

                    if (!seg.voice_type.empty()) {
                        VoicePreset preset = load_voice_preset(seg.voice_type);
                        if (!preset.ref_audio_path.empty()) {
                            params.ref_audio_path = preset.ref_audio_path;
                            params.ref_text = preset.ref_text;
                        } else {
                            std::string inferred_lang;
                            std::string inferred_instruct;
                            infer_from_capcut_voice(seg.voice_type, inferred_lang, inferred_instruct);
                            if (!inferred_lang.empty()) params.language = inferred_lang;
                            if (!inferred_instruct.empty()) {
                                params.instruct = inferred_instruct;
                            } else if (looks_like_omnivoice_instruct(seg.voice_type)) {
                                params.instruct = seg.voice_type;
                            }
                        }
                    }

                    if (!ov.empty()) {
                        std::string err_ignored;
                        apply_voice_and_ref_options(ov, params, err_ignored);
                        apply_generation_options(ov, params);
                        if (auto spd = get_opt_number(ov, "speed")) {
                            params.speed = clamp_speed(static_cast<float>(*spd));
                        }
                    }

                    const auto gen_start = std::chrono::steady_clock::now();
                    omnivoice::Audio audio;
                    {
                        std::lock_guard<std::mutex> lock(gen_mutex);
                        audio = runtime.generate(params);
                    }
                    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - gen_start).count();

                    CapCutArtifact art;
                    art.bytes = encode_audio(audio.samples, audio.sample_rate, audio_format);
                    art.format = audio_format;
                    art.text = seg.text;
                    art.duration_ms = audio.sample_rate > 0
                        ? static_cast<int>((audio.samples.size() * 1000ULL) / static_cast<size_t>(audio.sample_rate))
                        : 0;

                    std::cerr << std::fixed << std::setprecision(3)
                              << "[capcut-task] id=" << task->id
                              << " seg=" << (idx + 1) << "/" << segments.size()
                              << " gen=" << gen_s << "s"
                              << " audio=" << (art.duration_ms / 1000.0) << "s"
                              << " size=" << art.bytes.size() << "\n";

                    produced.push_back(std::move(art));
                }

                std::lock_guard<std::mutex> lock(tasks_mutex);
                task->artifacts = std::move(produced);
                task->status = "succeed";
            } catch (const std::exception & e) {
                std::lock_guard<std::mutex> lock(tasks_mutex);
                task->error = e.what();
                task->status = "failed";
                std::cerr << "[capcut-task] failed id=" << task->id << " error=" << e.what() << "\n";
            }
        });

        set_json(res, {
            {"ret", "0"},
            {"errmsg", ""},
            {"data", {
                {"bind_id", bind_id},
                {"tasks", json::array({
                    {
                        {"id", task->id},
                        {"token", task->token},
                        {"bind_id", bind_id},
                        {"req_key", "sami_text_to_speech"},
                        {"task_version", "v3"},
                        {"status", "running"},
                    }
                })},
            }},
        });
    });

    // POST /lv/v1/common_task/query — CapCut common-task compatible TTS status query
    svr.Post("/lv/v1/common_task/query", [&](const httplib::Request & req, httplib::Response & res) {
        if (req.body.empty()) {
            res.status = 400;
            res.set_content(json_error("request body is empty"), "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::parse_error & e) {
            res.status = 400;
            res.set_content(json_error(std::string("invalid JSON: ") + e.what()), "application/json");
            return;
        }
        if (!body.is_object() || !body.contains("tasks") || !body.at("tasks").is_array()
            || body.at("tasks").empty() || !body.at("tasks").at(0).is_object()) {
            set_json(res, {{"ret", "4000"}, {"errmsg", "missing tasks[0] in request body"}, {"data", json::object()}});
            return;
        }

        const json & q = body.at("tasks").at(0);
        const std::string req_key = get_opt_string(q, "req_key");
        if (req_key == "cc_audio_subtitle_asr") {
            set_json(res, {
                {"ret", "4001"},
                {"errmsg", "STT is not supported by omnivoice-server (TTS only). Use the CapCut backend for STT."},
                {"data", json::object()},
            });
            return;
        }

        const std::string task_id = get_opt_string(q, "id");
        const std::string token = get_opt_string(q, "token");
        const std::string req_bind_id = get_opt_string(q, "bind_id");

        std::shared_ptr<CapCutTask> task;
        std::string status;
        std::string error;
        std::string task_bind_id;
        std::vector<std::tuple<std::string, std::string, int>> art_meta;
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            auto it = capcut_tasks.find(task_id);
            if (it == capcut_tasks.end() || it->second->token != token) {
                set_json(res, {{"ret", "2001"}, {"errmsg", "task not found"}, {"data", json::object()}});
                return;
            }
            task = it->second;
            status = task->status;
            error = task->error;
            task_bind_id = req_bind_id.empty() ? task->bind_id : req_bind_id;
            if (status == "succeed") {
                art_meta.reserve(task->artifacts.size());
                for (const auto & art : task->artifacts) {
                    art_meta.emplace_back(art.format, art.text, art.duration_ms);
                }
            }
        }

        std::string host = req.get_header_value("X-Forwarded-Host");
        if (host.empty()) host = req.get_header_value("Host");
        if (host.empty()) host = "localhost:" + std::to_string(cfg.port);
        std::string proto = req.get_header_value("X-Forwarded-Proto");
        if (proto.empty()) proto = "http";
        const std::string origin = proto + "://" + host;

        json task_entry = {
            {"id", task->id},
            {"token", task->token},
            {"bind_id", task_bind_id},
            {"req_key", "sami_text_to_speech"},
            {"task_version", "v3"},
            {"status", status},
        };

        if (status == "succeed") {
            json audio_subtitles = json::array();
            for (size_t i = 0; i < art_meta.size(); ++i) {
                const auto & [fmt, text, dur_ms] = art_meta[i];
                const std::string artifact_url = origin + "/lv/v1/tts_artifact/" + task->id + "/" + std::to_string(i) + "." + fmt;
                audio_subtitles.push_back({
                    {"url", artifact_url},
                    {"speech_url", artifact_url},
                    {"audio_url", artifact_url},
                    {"text", text},
                    {"duration_ms", dur_ms},
                });
            }
            task_entry["payload"] = json{{"audio_subtitles", audio_subtitles}}.dump();
        } else if (status == "failed") {
            task_entry["errmsg"] = error;
            task_entry["payload"] = json{{"error", error}}.dump();
        }

        set_json(res, {
            {"ret", "0"},
            {"errmsg", ""},
            {"data", {
                {"bind_id", task_bind_id},
                {"tasks", json::array({task_entry})},
            }},
        });
    });

    // GET /lv/v1/tts_artifact/:task_id/:filename — Download synthesized segment audio
    svr.Get("/lv/v1/tts_artifact/:task_id/:filename", [&](const httplib::Request & req, httplib::Response & res) {
        const auto id_it = req.path_params.find("task_id");
        const auto fn_it = req.path_params.find("filename");
        if (id_it == req.path_params.end() || fn_it == req.path_params.end()) {
            res.status = 404;
            res.set_content(json_error("artifact not found", "not_found"), "application/json");
            return;
        }
        const std::string & task_id = id_it->second;
        const std::string & filename = fn_it->second;

        size_t index = 0;
        try {
            const size_t dot = filename.find('.');
            const std::string idx_str = (dot == std::string::npos) ? filename : filename.substr(0, dot);
            if (idx_str.empty()) throw std::runtime_error("empty index");
            size_t parsed = 0;
            const unsigned long val = std::stoul(idx_str, &parsed);
            if (parsed != idx_str.size()) throw std::runtime_error("invalid index");
            index = static_cast<size_t>(val);
        } catch (const std::exception &) {
            res.status = 404;
            res.set_content(json_error("invalid artifact filename", "not_found"), "application/json");
            return;
        }

        std::string bytes;
        std::string format;
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            auto it = capcut_tasks.find(task_id);
            if (it == capcut_tasks.end() || it->second->status != "succeed" || index >= it->second->artifacts.size()) {
                res.status = 404;
                res.set_content(json_error("artifact not found", "not_found"), "application/json");
                return;
            }
            bytes = it->second->artifacts[index].bytes;
            format = it->second->artifacts[index].format;
        }

        res.status = 200;
        res.set_content(bytes, format_content_type(format));
    });

    // --- Start ---
    std::cerr << "[init] listening on " << cfg.host << ":" << cfg.port << "\n";
    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "error: failed to listen on " << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }
    return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
