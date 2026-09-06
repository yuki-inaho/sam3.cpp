# Example executables for sam3.cpp

# Quantization tool (no GUI dependency)
add_executable(sam3_quantize quantize.cpp)
target_link_libraries(sam3_quantize PRIVATE ggml)

add_executable(sam3_benchmark benchmark.cpp)
target_link_libraries(sam3_benchmark PRIVATE sam3)

add_executable(sam3_profile_edgetam profile_edgetam.cpp)
target_link_libraries(sam3_profile_edgetam PRIVATE sam3)

# Headless segmentation CLI (no GUI, no video deps)
add_executable(sam3_seg sam3_seg.cpp)
target_link_libraries(sam3_seg PRIVATE sam3)

# SDL2 + ImGui are optional — build only if SDL2 AND OpenGL are found.
find_package(SDL2 QUIET)

if(SDL2_FOUND)
    find_package(OpenGL QUIET)
    if(OpenGL_FOUND)
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    add_executable(sam3_video main_video.cpp)
    target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
    else()
        message(STATUS "OpenGL not found — skipping GUI examples")
    endif()
else()
    message(STATUS "SDL2 not found — skipping GUI examples")
endif()
