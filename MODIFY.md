## Summary
Add an actual bitstream file (`.bin`) writing pipeline.

## Changes
- [NEW] `./src/cpp` - Copied CPP rANS entropy coder from DCVC-RT repo.
- [NEW] `./src/models/entropy_models.py` - Copied from DCVC-RT repo, removed CUDA ops.
- [NEW] `./src/utils/stream_helper.py` - Copied from DCVC-RT repo, for bitstream io usage.
- [ADD] `./src/models/common_model.py` - Arrtibute `cuda_streams`.
- [ADD] `./src/models/common_model.py` - Arrtibute `entropy_coder`, `gaussian_encoder`.
- [ADD] `./src/models/common_model.py` - Method `get_cuda_stream`.
- [ADD] `./src/models/common_model.py` - Method `update`, `set_use_two_entropy_coders`.
- [ADD] `./src/models/common_model.py` - Method `encode_z_index`, `decode_z_index` for z-index bitstream.
- [ADD] `./src/models/common_model.py` - Method `compress_four_part_prior`, `decompress_four_part_prior` for image encode/decode.
- [ADD] `./src/models/common_model.py` - Static method `single_part_for_writing_4x` for image encode/decode.
- [ADD] `./src/models/image_model.py` - Method `compress`, `decompress` for image encode/decode.
- [ADD] `./src/models/common_model.py` - Method `compress_dual_prior`, `decompress_dual_prior` for video encode/deccode.
- [ADD] `./src/models/common_model.py` - Static method `single_part_for_writing_2x` for video encode/decode.
- [ADD] `./src/models/video_model.py` - Method `compress`, `decompress` for video encode/decode.
- [ADD] `./test_video.py` - Args `--write_stream`, set to `True` to enable stream writing.
- [ADD] `./test_video.py` - Function `run_one_point_with_stream`, test with stream.

## Legacy Code Notes
- `./src/cpp` - Copied directly from DCVC-RT. Contains some Z(hyper information)-related encode/decode code in the cpp/pybind files.