#include "sam31_native.h"
#include "runtime.h"
#include "synthetic.h"
#include <cerrno>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#ifdef __linux__
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace sam31::native;
namespace {
size_t checks = 0;
void check(bool value, const std::string &message) {
    ++checks;
    if (!value) throw std::runtime_error("test assertion: " + message);
}
void rejects(const std::function<void()> &run, const std::string &name) {
    bool rejected = false;
    try { run(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "must reject " + name);
}
double mae(const Result &a, const Result &b) {
    check(a.width == b.width && a.height == b.height &&
              a.objects.size() == b.objects.size(), "comparable results");
    double sum = 0;
    size_t count = 0;
    for (size_t o = 0; o < a.objects.size(); ++o)
        for (size_t n = 0; n < a.objects[o].logits.size(); ++n) {
            sum += std::abs(double(a.objects[o].logits[n]) - b.objects[o].logits[n]);
            ++count;
        }
    return sum / count;
}
void valid(const Result &r, size_t frame, const std::vector<int64_t> &ids) {
    check(r.frame_index == frame && r.width == 80 && r.height == 64,
          "frame index and output resolution");
    check(r.objects.size() == ids.size(), "object count");
    for (size_t o = 0; o < ids.size(); ++o) {
        const auto &mask = r.objects[o];
        check(mask.object_id == ids[o], "stable 64-bit object ID and order");
        check(mask.logits.size() == r.width * r.height, "mask length");
        check(std::isfinite(mask.score) && std::isfinite(mask.quality), "finite scores");
        check(std::all_of(mask.logits.begin(), mask.logits.end(),
                          [](float v) { return std::isfinite(v); }), "finite logits");
        const auto range = std::minmax_element(mask.logits.begin(), mask.logits.end());
        check(*range.second - *range.first > 1e-8f, "nonconstant untrained mask logits");
    }
}
bool disable_exec() {
#ifdef __linux__
    // These restrictions apply before model construction. Native inference must still run.
    // Thread creation is intentionally allowed for OpenMP and sanitizers.
    sock_filter rules[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_execve, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#ifdef __NR_execveat
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_execveat, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
    };
    sock_fprog program{static_cast<unsigned short>(sizeof(rules) / sizeof(rules[0])), rules};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0)
        throw std::runtime_error("cannot install no-exec test sandbox");
    char path[] = "/bin/false";
    char *args[] = {path, nullptr};
    char *env[] = {nullptr};
    errno = 0;
    check(syscall(__NR_execve, path, args, env) == -1 && errno == EPERM,
          "kernel rejects execve of another executable");
    setenv("SAM31_PYTHON", "/NOT_A_PROGRAM", 1);
    setenv("SAM31_SCRIPT", "/NOT_A_SCRIPT", 1);
    setenv("PATH", "/NO_EXECUTABLES", 1);
    return true;
#else
    return false;
#endif
}
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        Json report;
        report["execve_denied_by_seccomp"] = disable_exec();
        rejects([&] { Model denied(argv[1]); }, "fixture without explicit permission");
        Options options;
        options.allow_test_fixture = true;
        options.total_frames = 6;
        rejects([&] { Model missing("/missing/sam31.gguf", options); }, "missing checkpoint");
        rejects([&] { Tensor too_large(size_t(1) << 30, 2, 1); }, "excess allocation before allocation");
        Model model(argv[1], options);
        const std::vector<Point> points{{1, .25f, .32f}, {2, .73f, .68f}};
        const auto image = synthetic_image(0);
        Session session(model);
        rejects([&] { session.step(image); }, "step before begin");
        rejects([&] { session.begin(image, {}); }, "no prompts");
        rejects([&] { session.begin(image, {{1, .3f, .4f, 0}}); }, "negative-only prompt");
        rejects([&] { session.begin(image, {{0, .3f, .4f}}); }, "zero ID");
        rejects([&] { session.begin(image, {{1, -0.01f, .4f}}); }, "negative coordinate");
        rejects([&] { session.begin(image, {{1, 1.01f, .4f}}); }, "coordinate beyond one");
        rejects([&] { session.begin(image, {{1, std::numeric_limits<float>::quiet_NaN(), .4f}}); }, "NaN coordinate");
        rejects([&] { session.begin(image, {{1, .3f, .4f, 2}}); }, "invalid point label");
        auto broken = image;
        broken.rgb.pop_back();
        rejects([&] { session.begin(broken, points); }, "truncated image bytes");
        check(Json::parse(session.stats_json()).at("frames").integer() == 0,
              "invalid begin leaves state uninitialized");
        std::vector<Point> many;
        for (int i = 0; i < 17; ++i) many.push_back({i + 1, .2f + .03f * i, .5f});
        rejects([&] { session.begin(image, many); }, "17 objects");
        std::vector<Result> baseline;
        baseline.push_back(session.begin(image, points));
        valid(baseline[0], 0, {1, 2});
        rejects([&] { session.begin(image, points); }, "begin twice");
        const auto before_invalid = session.stats_json();
        broken = image;
        ++broken.width;
        rejects([&] { session.step(broken); }, "resolution change");
        broken = image;
        broken.rgb.clear();
        rejects([&] { session.step(broken); }, "truncated next frame");
        check(session.stats_json() == before_invalid, "failed step commits no state");
        for (size_t i = 1; i < 6; ++i) {
            baseline.push_back(session.step(synthetic_image(i)));
            valid(baseline.back(), i, {1, 2});
        }
        auto stats = Json::parse(session.stats_json());
        check(stats.at("image_encoder").integer() == 6, "one image encoder call per frame");
        check(stats.at("vit_blocks").integer() == 32 * 6, "all 32 ViT blocks per frame");
        check(stats.at("interactive_head").integer() == 2, "initial per-object interactive heads");
        check(stats.at("multiplex_head").integer() == 5, "one multiplex head per propagation frame");
        check(stats.at("memory_encoder").integer() == 6, "predictions enter new memories");
        check(stats.at("memory_blocks").integer() == 4 * 5, "four memory blocks per propagation frame");
        report["six_frame_stats"] = stats;
        auto info = Json::parse(model.info_json());
        check(info.at("required_tensors").integer() == 867 &&
              info.at("used_tensors").integer() == 867, "all required learned parameters used");
        report["model"] = info;

        session.reset();
        check(Json::parse(session.stats_json()).at("frames").integer() == 0, "reset clears state");
        check(mae(session.begin(image, points), baseline[0]) == 0, "bitwise repeatable initial frame");
        for (size_t i = 1; i < 6; ++i)
            check(mae(session.step(synthetic_image(i)), baseline[i]) == 0, "bitwise repeatable temporal sequence");
        report["reset_sequence_bit_exact"] = true;

        Session changed_image(model);
        auto inverted = image;
        for (auto &v : inverted.rgb) v = uint8_t(255 - v);
        const double image_delta = mae(changed_image.begin(inverted, points), baseline[0]);
        check(image_delta > 1e-7, "image pixels affect output");
        report["image_intervention_logit_mae"] = image_delta;
        Session changed_point(model);
        auto moved_points = points;
        moved_points[0].x = .82f;
        moved_points[0].y = .12f;
        const double point_delta = mae(changed_point.begin(image, moved_points), baseline[0]);
        check(point_delta > 1e-7, "prompt coordinates affect output");
        report["point_intervention_logit_mae"] = point_delta;
        Session changed_memory(model);
        changed_memory.begin(image, points);
        changed_memory.zero_memory_for_test();
        const double memory_delta = mae(changed_memory.step(synthetic_image(1)), baseline[1]);
        check(memory_delta > 1e-7, "previous-frame memory affects propagation");
        check(Json::parse(changed_memory.stats_json()).at("memory_blocks").integer() == 4,
              "memory intervention does not bypass attention");
        report["memory_intervention_logit_mae"] = memory_delta;

        Options long_options = options;
        long_options.total_frames = 20;
        Model long_model(argv[1], long_options);
        Session long_session(long_model);
        long_session.begin(image, points);
        for (size_t i = 1; i < 20; ++i) {
            valid(long_session.step(synthetic_image(i)), i, {1, 2});
            check(Json::parse(long_session.stats_json()).at("memory_frames_stored").integer() <= 16,
                  "bounded initial plus recent memory frames");
        }
        check(Json::parse(long_session.stats_json()).at("memory_frames_stored").integer() == 16,
              "20-frame run exercises eviction");
        report["twenty_frame_stats"] = Json::parse(long_session.stats_json());

        Session all_slots(model);
        many.pop_back();
        many[15].object_id = std::numeric_limits<int64_t>::max();
        std::vector<int64_t> all_ids;
        for (const auto &p : many) all_ids.push_back(p.object_id);
        valid(all_slots.begin(image, many), 0, all_ids);
        valid(all_slots.step(synthetic_image(1)), 1, all_ids);
        report["sixteen_objects_and_int64_ids"] = true;
        Session neg_points(model);
        auto positive_negative = points;
        positive_negative.push_back({1, .05f, .05f, 0});
        valid(neg_points.begin(image, positive_negative), 0, {1, 2});
        report["negative_prompt_supported"] = true;

        Options low_options = options;
        low_options.weight_cache_mb = 1;
        Model low_cache(argv[1], low_options);
        Session low_session(low_cache);
        check(mae(low_session.begin(image, points), baseline[0]) == 0, "cache eviction preserves initial output");
        check(mae(low_session.step(synthetic_image(1)), baseline[1]) == 0, "cache eviction preserves propagation");
        report["one_mib_cache_bit_exact"] = true;
        report["checks"] = checks;
        report["passed"] = true;
        report["python_inference"] = false;
        report["accuracy_validation_performed"] = false;
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "native graph test failed: " << e.what() << '\n';
        return 1;
    }
}
