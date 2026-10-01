#include "sam31_gguf.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#ifndef SAM31_SCRIPT_PATH
#error SAM31_SCRIPT_PATH is required
#endif
#ifndef SAM31_PYTHON_DEFAULT
#define SAM31_PYTHON_DEFAULT "python3"
#endif

namespace {
void usage() {
    std::cout << "sam31 inspect MODEL.gguf [--allow-test-fixture] [--read-all]\n"
                 "sam31 image --model MODEL.gguf --source-root CPU_SOURCE --image INPUT --point ID:X:Y --output DIR\n"
                 "sam31 video --model MODEL.gguf --source-root CPU_SOURCE --frames DIR --point ID:X:Y --output DIR\n"
                 "sam31 doctor|synthetic|dod|prepare-source --help\n\n"
                 "Storage and ConvRot primitives are native C++. Image/video inference uses the\n"
                 "explicit official-PyTorch reference backend; it is NOT a native ggml SAM 3.1 graph.\n"
                 "SAM31_PYTHON and SAM31_SCRIPT may override the interpreter and entry script.\n";
}

int inspect(int argc, char ** argv) {
    if (argc < 3) throw std::invalid_argument("inspect requires a GGUF file");
    bool allow_fixture = false, read_all = false;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--allow-test-fixture") allow_fixture = true;
        else if (option == "--read-all") read_all = true;
        else throw std::invalid_argument("Unknown inspect option: " + option);
    }
    sam31::gguf_file file(argv[2], allow_fixture);
    uint64_t bytes = 0;
    size_t int8_tensors = 0;
    for (const auto & item : file.tensors()) {
        bytes += item.size;
        int8_tensors += item.type == 24;
        if (read_all) (void) file.read_tensor(item.name);
    }
    std::cout << "{\n  \"architecture\": \"sam3_1_multiplex\",\n  \"storage_version\": 1,\n"
              << "  \"tensor_count\": " << file.tensors().size() << ",\n"
              << "  \"i8_storage_tensors\": " << int8_tensors << ",\n"
              << "  \"payload_bytes\": " << bytes << ",\n"
              << "  \"data_offset\": " << file.data_offset() << ",\n"
              << "  \"source_sha256\": \"" << file.source_sha256() << "\",\n"
              << "  \"test_fixture\": " << (file.is_test_fixture() ? "true" : "false") << ",\n"
              << "  \"all_payloads_read\": " << (read_all ? "true" : "false") << ",\n"
              << "  \"native_sam31_inference\": false\n}\n";
    return 0;
}

int run_reference(int argc, char ** argv, const char * fixed_command = nullptr) {
    const char * python = std::getenv("SAM31_PYTHON");
    const char * script = std::getenv("SAM31_SCRIPT");
    if (!python || !*python) python = SAM31_PYTHON_DEFAULT;
    if (!script || !*script) script = SAM31_SCRIPT_PATH;
    std::vector<std::string> arguments{python, script};
    if (fixed_command) arguments.emplace_back(fixed_command);
    for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
    std::vector<char *> raw;
    for (auto & arg : arguments) raw.push_back(const_cast<char *>(arg.c_str()));
    raw.push_back(nullptr);
    // No command shell: file paths and prompt strings are not interpolated.
#ifdef _WIN32
    const intptr_t status = _spawnvp(_P_WAIT, python, raw.data());
    if (status != -1) return static_cast<int>(status);
#else
    execvp(python, raw.data());
#endif
    throw std::runtime_error(std::string("Could not start Python reference backend: ") + std::strerror(errno));
}
}

int main(int argc, char ** argv) {
    try {
#ifdef SAM31_FIXED_COMMAND
        return run_reference(argc, argv, SAM31_FIXED_COMMAND);
#else
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            usage();
            return argc < 2 ? 2 : 0;
        }
        const std::string command = argv[1];
        if (command == "inspect") return inspect(argc, argv);
        if (command == "image" || command == "video" || command == "doctor" ||
            command == "synthetic" || command == "dod" || command == "prepare-source")
            return run_reference(argc, argv);
        throw std::invalid_argument("Unknown command: " + command);
#endif
    } catch (const std::exception & error) {
        std::cerr << "sam31: " << error.what() << '\n';
        return 2;
    }
}
