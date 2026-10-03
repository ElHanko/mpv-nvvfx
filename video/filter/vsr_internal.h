/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#pragma once

#include <stdbool.h>
#include "cuda_shared.h"

// C ABI declarations derived from the reference, without SDK binaries/headers.
typedef struct NvCVImage {
    unsigned int width, height;
    int pitch;
    int pixelFormat, componentType;
    unsigned char pixelBytes, componentBytes, numComponents;
    unsigned char planar, gpuMem, colorspace;
    void *pixels;
    void *deletePtr;
    void (*deleteProc)(void *);
    unsigned long long bufferBytes;
} NvCVImage;

struct vsr_runtime {
    void *libs[20];
    int num_libs;
    char *dir;
    int (*create)(const char *, void **);
    void (*destroy)(void *);
    int (*set_u32)(void *, const char *, unsigned int);
    int (*set_image)(void *, const char *, NvCVImage *);
    int (*set_stream)(void *, const char *, CUstream);
    int (*load)(void *);
    int (*run)(void *, int);
    int (*alloc)(NvCVImage *, unsigned int, unsigned int,
                 int, int, int, int, unsigned int);
    void (*dealloc)(NvCVImage *);
};

struct vsr_context {
    void *effect;
    NvCVImage input, output;
    CUstream stream;
};

bool vsr_runtime_load(struct vsr_runtime *r, struct mp_log *log);
void vsr_runtime_unload(struct vsr_runtime *r);
bool vsr_init(struct vsr_context *c, struct vsr_runtime *r,
              struct mp_log *log, int in_w, int in_h, int out_w, int out_h,
              int quality, CUstream stream);
bool vsr_process(struct vsr_context *c, struct vsr_runtime *r,
                 struct mp_log *log);
// Caller must make the borrowed CUDA context current and finish stream work.
void vsr_destroy(struct vsr_context *c, struct vsr_runtime *r);
