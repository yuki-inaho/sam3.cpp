# SAM 3.1 native C++ CPU

See [the complete Japanese guide](../docs/SAM31_GUIDE.ja.md) for checkpoint acquisition, lossless ConvRot SafeTensors → GGUF conversion, build, image/video annotation, validation and archive restoration.

Normal inference uses C++ and FP32 CPU/CBLAS. Python/NumPy is used for storage conversion and test drivers. The optional Python reference launcher is disabled by default. `TEST_ONLY` fixtures are untrained; actual checkpoint validation is recorded separately under `evidence/` in the source delivery.
