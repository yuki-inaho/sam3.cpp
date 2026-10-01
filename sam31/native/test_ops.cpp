#include "runtime.h"
#include <fstream>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace sam31::native;
namespace {
Tensor unpack(const Json &j) {
    const auto &s = j.at("shape").arr();
    require(s.size() == 3, "operator oracle must use HWC rank three");
    Tensor value(s[0].integer(), s[1].integer(), s[2].integer());
    const auto &data = j.at("values").arr();
    require(data.size() == value.v.size(), "oracle element count");
    for (size_t i = 0; i < data.size(); ++i) value.v[i] = float(data[i].number());
    return value;
}
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        std::ifstream file(argv[2]);
        require(bool(file), "cannot open independent operator fixture");
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        const auto oracle = Json::parse(text);
        Weights weights(argv[1], true, 128);
        require(weights.hash() == oracle.at("source_sha256").str(), "oracle/checkpoint hash mismatch");
        Counters calls;
        Ops ops(weights, calls);
        size_t checks = 0;
        Json report;
        Json::array results;
        for (const auto &c : oracle.at("cases").arr()) {
            std::vector<Tensor> inputs;
            for (const auto &input : c.at("inputs").arr()) inputs.push_back(unpack(input));
            const auto &x = inputs.at(0);
            const auto &op = c.at("operation").str();
            Tensor actual;
            if (op == "linear") actual = ops.linear(x, c.at("prefix").str());
            else if (op == "norm") actual = ops.norm(x, c.at("prefix").str(), 1e-6f);
            else if (op == "conv") actual = ops.conv(x, c.at("prefix").str(), c.at("stride").integer(), c.at("padding").integer(), c.at("bias").boolean(), c.at("depthwise").boolean());
            else if (op == "deconv") actual = ops.deconv(x, c.at("prefix").str());
            else if (op == "resize") actual = resize(x, c.at("height").integer(), c.at("width").integer(), c.at("antialias").boolean());
            else if (op == "attention") actual = attention(x, inputs.at(1), inputs.at(2), c.at("heads").integer());
            else if (op == "rope") { actual = x; rope(actual, c.at("heads").integer(), c.at("grid").integer(), 1, c.at("exclude").integer()); }
            else if (op == "gelu") actual = gelu(x);
            else throw std::runtime_error("unknown operator oracle operation");
            const Tensor expected = unpack(c.at("output"));
            require(actual.h == expected.h && actual.w == expected.w && actual.c == expected.c, "operator output shape mismatch");
            double max_error = 0;
            for (size_t i = 0; i < actual.v.size(); ++i) {
                const double error = std::abs(double(actual.v[i]) - expected.v[i]);
                max_error = std::max(error, max_error);
                require(std::isfinite(actual.v[i]) && error <= oracle.at("atol").number() + oracle.at("rtol").number() * std::abs(expected.v[i]),
                        "operator mismatch: " + c.at("name").str() + " element " + std::to_string(i) + " abs error " + std::to_string(error));
                ++checks;
            }
            Json item;
            item["name"] = c.at("name").str(); item["max_abs_error"] = max_error;
            item["elements"] = actual.v.size();
            results.push_back(item);
        }
        report["cases"] = results;
        report["element_checks"] = checks;
        report["passed"] = true;
        report["full_official_model_equivalence_test"] = false;
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native operator test: " << error.what() << '\n';
        return 1;
    }
}
