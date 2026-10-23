# JEV (Tev1 decision model) on llama.cpp, CPU only.
# llama.cpp is fetched at a pinned release. To build from a local checkout instead (offline builds):
#   gradlew assembleDebug -PllamaSource=E:/path/to/llama.cpp
cmake_minimum_required(VERSION 3.22)
project(jev LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# llama.cpp is unusably slow unoptimised: always Release, also inside a debug build of the app.
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(LLAMA_TAG b11800)
set(LLAMA_SOURCE "" CACHE PATH "local llama.cpp checkout (empty: download ${LLAMA_TAG})")

# Plain CPU backend, no tools or tests.
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
if(ANDROID)
    # Every ARM CPU variant (plain ARMv8, dot product, int8 matrix multiply, SVE...) as its own library;
    # the phone loads the best one its CPU supports (JevModel::load_backends), like llama.cpp's own
    # Android example. int8 matrix multiply roughly doubles the speed of the Q8_0 model on recent phones.
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "" CACHE STRING "" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()

if(LLAMA_SOURCE)
    add_subdirectory(${LLAMA_SOURCE} llama-build)
else()
    include(FetchContent)
    FetchContent_Declare(llama
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG ${LLAMA_TAG}
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(llama)
endif()

if(ANDROID)
    add_library(jev SHARED jev_core.cpp jev_jni.cpp)
    target_link_libraries(jev PRIVATE llama ggml android log)
    target_compile_options(jev PRIVATE -O3)
else()
    add_library(jev_core STATIC jev_core.cpp)
    target_include_directories(jev_core PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    target_link_libraries(jev_core PUBLIC llama ggml)
endif()
