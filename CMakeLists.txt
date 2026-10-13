cmake_minimum_required(VERSION 3.14)
project(sam3.cpp VERSION 1.0.0 LANGUAGES C CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml options — enable Metal on Apple
option(SAM3_METAL "Enable Metal backend" ON)
if(APPLE AND SAM3_METAL)
    set(GGML_METAL        ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h efficient_sam2/sparse_policy.cpp efficient_sam2/resize.cpp efficient_sam2/jpeg_decode.cpp efficient_sam2/runtime_identity.cpp)
target_include_directories(sam3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)
find_package(JPEG REQUIRED)
target_link_libraries(sam3 PUBLIC ggml JPEG::JPEG ${CMAKE_DL_LIBS})
target_compile_features(sam3 PUBLIC cxx_std_14)

# Content-based implementation identity for persistent exact-prefix artifacts.
# Reconfigure when source changes, including uncommitted changes, rather than
# treating a Git commit ID as sufficient evidence of numerical compatibility.
file(GLOB_RECURSE SAM2_CACHE_ID_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/*.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/stb/*.h")
list(APPEND SAM2_CACHE_ID_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sam3.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/sam3.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sam31/src/quant.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sam31/include/sam31_quant.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sam31/native/json.h")
list(SORT SAM2_CACHE_ID_SOURCES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SAM2_CACHE_ID_SOURCES})
set(SAM2_CACHE_ID_INPUT "${CMAKE_CXX_COMPILER_ID}|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_C_COMPILER_ID}|${CMAKE_C_COMPILER_VERSION}|${CMAKE_BUILD_TYPE}|${CMAKE_CXX_FLAGS}|${CMAKE_CXX_FLAGS_RELEASE}|${CMAKE_C_FLAGS}|${CMAKE_C_FLAGS_RELEASE}|${CMAKE_SYSTEM_NAME}|${CMAKE_SYSTEM_PROCESSOR}")
get_cmake_property(SAM2_BUILD_VARIABLES VARIABLES)
list(SORT SAM2_BUILD_VARIABLES)
foreach(SAM2_VARIABLE IN LISTS SAM2_BUILD_VARIABLES)
    if(SAM2_VARIABLE MATCHES "^GGML_")
        string(APPEND SAM2_CACHE_ID_INPUT "|${SAM2_VARIABLE}=${${SAM2_VARIABLE}}")
    endif()
    if(SAM2_VARIABLE MATCHES "^CMAKE_(C|CXX)_FLAGS")
        string(APPEND SAM2_CACHE_ID_INPUT "|${SAM2_VARIABLE}=${${SAM2_VARIABLE}}")
    endif()
endforeach()
foreach(SAM2_SOURCE IN LISTS SAM2_CACHE_ID_SOURCES)
    file(SHA256 "${SAM2_SOURCE}" SAM2_SOURCE_HASH)
    file(RELATIVE_PATH SAM2_SOURCE_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SAM2_SOURCE}")
    string(APPEND SAM2_CACHE_ID_INPUT "|${SAM2_SOURCE_REL}:${SAM2_SOURCE_HASH}")
endforeach()
string(APPEND SAM2_CACHE_ID_INPUT "|JPEG_VERSION=${JPEG_VERSION}|JPEG_LIBRARY=${JPEG_LIBRARIES}")
foreach(SAM2_JPEG_LIBRARY IN LISTS JPEG_LIBRARIES)
    if(EXISTS "${SAM2_JPEG_LIBRARY}")
        file(SHA256 "${SAM2_JPEG_LIBRARY}" SAM2_JPEG_HASH)
        string(APPEND SAM2_CACHE_ID_INPUT "|jpeg_sha256=${SAM2_JPEG_HASH}")
    endif()
endforeach()
string(SHA256 SAM2_CACHE_BUILD_ID "${SAM2_CACHE_ID_INPUT}")
target_compile_definitions(sam3 PRIVATE SAM2_CACHE_BUILD_ID="${SAM2_CACHE_BUILD_ID}")
target_compile_definitions(sam3 PRIVATE SAM2_CACHE_CONFIGURATION="$<CONFIG>")
add_subdirectory(efficient_sam2/prefix_cache)
target_link_libraries(sam3 PRIVATE efficient_sam2_prefix_cache)

# Storage-only Efficient-SAM2 adapter. Reuse the existing regular-H4 reference
# primitives; the inference graph consumes restored FP32 weights explicitly.
add_library(efficient_sam2_storage STATIC
    efficient_sam2/gguf_loader.cpp sam31/src/quant.cpp)
target_include_directories(efficient_sam2_storage PRIVATE sam31/include)
target_compile_definitions(efficient_sam2_storage PRIVATE SAM31_QUANT_STATIC=1)
target_compile_features(efficient_sam2_storage PRIVATE cxx_std_17)
target_link_libraries(efficient_sam2_storage PUBLIC ggml)
target_link_libraries(sam3 PRIVATE efficient_sam2_storage)
add_executable(efficient_sam2_inspect efficient_sam2/inspect.cpp)
target_link_libraries(efficient_sam2_inspect PRIVATE efficient_sam2_storage)
add_executable(test_efficient_sam2_policy efficient_sam2/test_policy.cpp efficient_sam2/sparse_policy.cpp)
target_compile_features(test_efficient_sam2_policy PRIVATE cxx_std_14)
add_test(NAME efficient_sam2_sparse_policy COMMAND test_efficient_sam2_policy)
add_executable(test_efficient_sam2_fpn_cache efficient_sam2/test_fpn_cache.cpp)
target_compile_features(test_efficient_sam2_fpn_cache PRIVATE cxx_std_14)
add_test(NAME efficient_sam2_fpn_cache COMMAND test_efficient_sam2_fpn_cache)
add_executable(efficient_sam2_track efficient_sam2/track_cli.cpp)
target_link_libraries(efficient_sam2_track PRIVATE sam3)
target_compile_features(efficient_sam2_track PRIVATE cxx_std_17)
add_executable(test_efficient_sam2_deconv efficient_sam2/test_deconv.cpp)
target_link_libraries(test_efficient_sam2_deconv PRIVATE ggml)
add_test(NAME efficient_sam2_f32_deconv COMMAND test_efficient_sam2_deconv)
add_executable(test_efficient_sam2_resize efficient_sam2/resize.cpp efficient_sam2/test_resize.cpp)
add_executable(test_efficient_sam2_jpeg efficient_sam2/jpeg_decode.cpp efficient_sam2/test_jpeg_decode.cpp)
target_link_libraries(test_efficient_sam2_jpeg PRIVATE JPEG::JPEG)
add_executable(test_efficient_sam2_runtime_identity efficient_sam2/runtime_identity.cpp efficient_sam2/test_runtime_identity.cpp)
target_link_libraries(test_efficient_sam2_runtime_identity PRIVATE efficient_sam2_prefix_cache ${CMAKE_DL_LIBS})
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME efficient_sam2_runtime_identity COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/test_runtime_identity.py $<TARGET_FILE:test_efficient_sam2_runtime_identity>)
    endif()
    add_test(NAME efficient_sam2_storage_contract COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/test_storage.py
        $<TARGET_FILE:efficient_sam2_inspect>)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, PIL" RESULT_VARIABLE SAM2_HAS_RESIZE_ORACLE
        OUTPUT_QUIET ERROR_QUIET)
    if(SAM2_HAS_RESIZE_ORACLE EQUAL 0)
        add_test(NAME efficient_sam2_video_jpeg COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/test_jpeg_decode.py $<TARGET_FILE:test_efficient_sam2_jpeg>)
        add_test(NAME efficient_sam2_video_resize COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/efficient_sam2/test_resize.py $<TARGET_FILE:test_efficient_sam2_resize>)
    endif()
endif()

# Examples (only when top-level)
option(SAM3_BUILD_EXAMPLES "Build example executables" ON)
if(SAM3_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

# Tests
option(SAM3_BUILD_TESTS "Build test executables" OFF)
if(SAM3_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Separate SAM 3.1 Multiplex CPU / GGUF extension.
option(SAM3_BUILD_SAM31 "Build SAM 3.1 native image/video inference" ON)
if(SAM3_BUILD_SAM31)
    enable_testing()
    add_subdirectory(sam31)
endif()
