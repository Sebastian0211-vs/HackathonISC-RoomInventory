// JNI bridge for ch.hevs.roominventory.JevNative. Calls on one handle must come from one thread at a time.
#include <jni.h>

#include <android/log.h>

#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "jev_core.h"
#include "llama.h"

#define TAG "RoomInventory"

namespace {

std::once_flag backends_once;

// llama.cpp and ggml log to stderr, which Android drops: keep warnings, errors and the backend choice.
void log_to_logcat(ggml_log_level level, const char * text, void *) {
    std::string t(text);
    bool backend = t.find("backend") != std::string::npos || t.find("CPU") != std::string::npos;
    if (level == GGML_LOG_LEVEL_ERROR) __android_log_print(ANDROID_LOG_ERROR, TAG, "llama: %s", text);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, "llama: %s", text);
    else if (backend && level != GGML_LOG_LEVEL_DEBUG) __android_log_print(ANDROID_LOG_INFO, TAG, "llama: %s", text);
}

std::string to_string(JNIEnv * env, jstring s) {
    const char * chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

void throw_java(JNIEnv * env, const char * message, const char * cls_name = "java/lang/IllegalStateException") {
    jclass cls = env->FindClass(cls_name);
    env->ThrowNew(cls, message);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_ch_hevs_roominventory_JevNative_nativeLoad(JNIEnv * env, jclass, jstring libDir, jstring path, jint threads, jint ctx) {
    try {
        std::string dir = to_string(env, libDir);
        std::call_once(backends_once, [&] {
            llama_log_set(log_to_logcat, nullptr);
            JevModel::load_backends(dir);
        });
        auto * m = new JevModel(to_string(env, path), threads, ctx);
        __android_log_print(ANDROID_LOG_INFO, TAG, "JEV loaded: %s", m->describe().c_str());
        return reinterpret_cast<jlong>(m);
    } catch (const std::exception & e) {
        throw_java(env, e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_ch_hevs_roominventory_JevNative_nativeOptionLogits(JNIEnv * env, jclass, jlong handle, jstring prompt,
                                                         jobjectArray options) {
    auto * m = reinterpret_cast<JevModel *>(handle);
    try {
        std::vector<std::string> opts;
        jsize n = env->GetArrayLength(options);
        for (jsize i = 0; i < n; i++) {
            auto s = (jstring) env->GetObjectArrayElement(options, i);
            opts.push_back(to_string(env, s));
            env->DeleteLocalRef(s);
        }
        std::vector<float> logits = m->option_logits(to_string(env, prompt), opts);
        jfloatArray out = env->NewFloatArray((jsize) logits.size());
        env->SetFloatArrayRegion(out, 0, (jsize) logits.size(), logits.data());
        return out;
    } catch (const JevAborted & e) {
        throw_java(env, e.what(), "java/util/concurrent/CancellationException");
        return nullptr;
    } catch (const std::exception & e) {
        throw_java(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ch_hevs_roominventory_JevNative_nativeSetAbort(JNIEnv *, jclass, jboolean abort) {
    JevModel::set_abort(abort == JNI_TRUE);
}

extern "C" JNIEXPORT jint JNICALL
Java_ch_hevs_roominventory_JevNative_nativeLastPromptTokens(JNIEnv *, jclass, jlong handle) {
    return reinterpret_cast<JevModel *>(handle)->last_prompt_tokens();
}

extern "C" JNIEXPORT jstring JNICALL
Java_ch_hevs_roominventory_JevNative_nativeDescribe(JNIEnv * env, jclass, jlong handle) {
    return env->NewStringUTF(reinterpret_cast<JevModel *>(handle)->describe().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_ch_hevs_roominventory_JevNative_nativeFree(JNIEnv *, jclass, jlong handle) {
    delete reinterpret_cast<JevModel *>(handle);
}
