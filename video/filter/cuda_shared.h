/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#pragma once

#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>

#include "common/msg.h"
#include "common/common.h"
#include "video/mp_image.h"

// Borrow the decoder's context. No new device/context or host upload.
static inline AVCUDADeviceContext *vsr_cuda_device(struct mp_image *img)
{
    if (img->imgfmt != IMGFMT_CUDA || !img->hwctx)
        return NULL;
    AVHWFramesContext *frames = (void *)img->hwctx->data;
    if (frames->device_ctx->type != AV_HWDEVICE_TYPE_CUDA)
        return NULL;
    return frames->device_ctx->hwctx;
}

static inline bool vsr_cuda_check(struct mp_log *log, CudaFunctions *cu,
                                  CUresult result, const char *operation)
{
    if (result == CUDA_SUCCESS)
        return true;
    const char *message = NULL;
    cu->cuGetErrorString(result, &message);
    mp_err(log, "VSR: %s failed (%d): %s\n", operation, result,
           message ? message : "unknown CUDA error");
    return false;
}
