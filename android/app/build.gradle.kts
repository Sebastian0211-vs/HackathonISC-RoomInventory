import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ch.hevs.roominventory"
    compileSdk = 36
    // JEV runs on llama.cpp, built from source by src/main/cpp/CMakeLists.txt.
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "ch.hevs.roominventory"
        minSdk = 29 // MediaStore RELATIVE_PATH for saving crops without storage permission
        targetSdk = 36
        versionCode = 2
        versionName = "0.2"
        // Phones only: keeps the native libraries to a single ABI.
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake {
                // llama.cpp is unusably slow without optimisations, also in debug builds of the app
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                // offline builds: gradlew assembleDebug -PllamaSource=C:/path/to/llama.cpp
                (project.findProperty("llamaSource") as String?)?.let { arguments += "-DLLAMA_SOURCE=$it" }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    signingConfigs {
        // Committed debug key, so APKs from CI and from any laptop install over each other.
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // Models are memory-mapped or copied straight from the APK, so they must stay uncompressed.
    androidResources {
        noCompress += listOf("tflite", "gguf", "bin")
        // gradlew installDebug -PnoJev: an APK without the 0.8 GB JEV model, for quick reinstalls while
        // developing (the phone keeps the copy it unpacked from an earlier full install)
        if (project.hasProperty("noJev")) ignoreAssetsPatterns += "!jev.gguf"
    }

    // ggml loads its CPU variant libraries from the app's native library folder at run time,
    // so they must be extracted there at install (as in llama.cpp's Android example).
    packaging {
        jniLibs.useLegacyPackaging = true
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    val camerax = "1.5.3"
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.12.4")
    implementation("androidx.core:core-ktx:1.17.0")

    // LiteRT 2 CompiledModel API; litert-gpu adds the ML Drift GPU accelerator (OpenCL/OpenGL)
    val litert = "2.3.0"
    implementation("com.google.ai.edge.litert:litert:$litert")
    implementation("com.google.ai.edge.litert:litert-gpu:$litert")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303") // Android's org.json is a stub in JVM unit tests
}
