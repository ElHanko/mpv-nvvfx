/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#pragma once

#include "cuda_shared.h"

enum yuv_matrix {
    YUV_MATRIX_BT601,
    YUV_MATRIX_BT709,
    YUV_MATRIX_BT2020,
};

enum yuv_range {
    YUV_RANGE_LIMITED,
    YUV_RANGE_FULL,
};

struct yuv_to_rgba {
    CUmodule module;
    CUfunction kernel;
};

bool yuv_to_rgba_init(struct yuv_to_rgba *c, CudaFunctions *cu,
                      struct mp_log *log, const char *runtime_dir,
                      int bit_depth, enum yuv_matrix matrix, enum yuv_range range);
bool yuv_to_rgba_convert(struct yuv_to_rgba *c, CudaFunctions *cu,
                         struct mp_log *log, struct mp_image *input,
                         void *output, int output_pitch, CUstream stream);
void yuv_to_rgba_destroy(struct yuv_to_rgba *c, CudaFunctions *cu);
