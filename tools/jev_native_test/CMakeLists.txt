# Desktop build of the app's JEV code (android/app/src/main/cpp/jev_core.cpp), for tools/export_jev.py --check.
#   cmake -S tools/jev_native_test -B tools/jev_native_test/build -DLLAMA_SOURCE=<abs path>/tools/.work/llama.cpp
#   cmake --build tools/jev_native_test/build --config Release
cmake_minimum_required(VERSION 3.22)
project(jev_native_test LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../android/app/src/main/cpp jev)
add_executable(jev_test main.cpp)
target_link_libraries(jev_test PRIVATE jev_core)
