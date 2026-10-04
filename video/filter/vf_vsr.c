/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#include "vsr_internal.h"
#include "yuv_to_rgba.h"

#include <math.h>
#include <libavutil/frame.h>
#include <libavutil/rational.h>

#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"

struct vsr_opts {
    double scale;
    int quality;
};

struct priv {
    struct vsr_opts *opts;
    struct vsr_runtime runtime;
    CudaFunctions *cu;
    CUcontext ctx;
    CUstream stream;
    CUevent decoded;
    AVBufferRef *input_frames;
    AVBufferRef *output_frames;
    struct vsr_context vsr;
    struct yuv_to_rgba converter;
    struct mp_image_params input_params;
    bool configured;
};

#define CUDA_CHECK(f, expression) \
    vsr_cuda_check((f)->log, p->cu, (expression), #expression)

static void release_pipeline(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (p->ctx) {
        CUcontext dummy;
        if (CUDA_CHECK(f, p->cu->cuCtxPushCurrent(p->ctx))) {
            // Wait only for our own work, never the VO's stream/default context.
            if (p->stream)
                CUDA_CHECK(f, p->cu->cuStreamSynchronize(p->stream));
            vsr_destroy(&p->vsr, &p->runtime);
            yuv_to_rgba_destroy(&p->converter, p->cu);
            if (p->decoded)
                CUDA_CHECK(f, p->cu->cuEventDestroy(p->decoded));
            if (p->stream)
                CUDA_CHECK(f, p->cu->cuStreamDestroy(p->stream));
            CUDA_CHECK(f, p->cu->cuCtxPopCurrent(&dummy));
        }
    }
    av_buffer_unref(&p->output_frames);
    // Release last: output frame contexts independently hold the same device.
    av_buffer_unref(&p->input_frames);
    p->ctx = NULL;
    p->stream = NULL;
    p->decoded = NULL;
    p->configured = false;
}

static bool configure(struct mp_filter *f, struct mp_image *input,
                       int out_w, int out_h, enum yuv_matrix matrix,
                       enum yuv_range range)
{
    struct priv *p = f->priv;
    if (!CUDA_CHECK(f, p->cu->cuStreamCreate(&p->stream, CU_STREAM_NON_BLOCKING)) ||
        !CUDA_CHECK(f, p->cu->cuEventCreate(&p->decoded, CU_EVENT_DISABLE_TIMING)))
        return false;

    AVHWFramesContext *source = (void *)p->input_frames->data;
    p->output_frames = av_hwframe_ctx_alloc(source->device_ref);
    if (!p->output_frames)
        return false;
    AVHWFramesContext *frames = (void *)p->output_frames->data;
    frames->format = AV_PIX_FMT_CUDA;
    frames->sw_format = AV_PIX_FMT_RGBA;
    frames->width = out_w;
    frames->height = out_h;
    int result = av_hwframe_ctx_init(p->output_frames);
    if (result < 0) {
        MP_ERR(f, "VSR: cannot initialize CUDA/RGBA output frames (%d).\n", result);
        return false;
    }
    int bit_depth = input->params.hw_subfmt == IMGFMT_NV12 ? 8 : 16;
    if (!yuv_to_rgba_init(&p->converter, p->cu, f->log, p->runtime.dir,
                          bit_depth, matrix, range) ||
        !vsr_init(&p->vsr, &p->runtime, f->log, input->w, input->h,
                  out_w, out_h, p->opts->quality, p->stream))
        return false;
    p->input_params = input->params;
    p->configured = true;
    return true;
}

static void process(struct mp_filter *f)
{
    struct priv *p = f->priv;
    if (!mp_pin_can_transfer_data(f->ppins[1], f->ppins[0]))
        return;
    struct mp_frame frame = mp_pin_out_read(f->ppins[0]);
    if (mp_frame_is_signaling(frame)) {
        mp_pin_in_write(f->ppins[1], frame);
        return;
    }
    if (frame.type != MP_FRAME_VIDEO)
        goto fail;

    struct mp_image *input = frame.data;
    AVCUDADeviceContext *device = vsr_cuda_device(input);
    if (!device || (input->params.hw_subfmt != IMGFMT_NV12 &&
                    input->params.hw_subfmt != IMGFMT_P010)) {
        MP_ERR(f, "VSR: requires CUDA NV12/P010 frames; use --hwdec=nvdec "
               "with --vo=gpu-next and CUDA interop.\n");
        goto fail;
    }
    int out_w = lrint(input->w * p->opts->scale);
    int out_h = lrint(input->h * p->opts->scale);
    if (out_w <= 0 || out_h <= 0 || out_w > 8192 || out_h > 8192) {
        MP_ERR(f, "VSR: scaled dimensions %dx%d exceed the MVP limit (8192).\n",
               out_w, out_h);
        goto fail;
    }

    enum pl_color_system system = input->params.repr.sys;
    if (system == PL_COLOR_SYSTEM_UNKNOWN)
        system = mp_csp_guess_colorspace(input->w, input->h);
    enum yuv_matrix matrix;
    switch (system) {
    case PL_COLOR_SYSTEM_BT_601: matrix = YUV_MATRIX_BT601; break;
    case PL_COLOR_SYSTEM_BT_709: matrix = YUV_MATRIX_BT709; break;
    case PL_COLOR_SYSTEM_BT_2020_NC: matrix = YUV_MATRIX_BT2020; break;
    default:
        MP_ERR(f, "VSR: unsupported YUV color system %d.\n", system);
        goto fail;
    }
    enum yuv_range range = input->params.repr.levels == PL_COLOR_LEVELS_FULL ?
                            YUV_RANGE_FULL : YUV_RANGE_LIMITED;

    if (p->ctx && (p->ctx != device->cuda_ctx ||
        p->input_params.w != input->w || p->input_params.h != input->h ||
        p->input_params.hw_subfmt != input->params.hw_subfmt ||
        p->input_params.repr.sys != input->params.repr.sys ||
        p->input_params.repr.levels != input->params.repr.levels))
        release_pipeline(f);
    if (!p->ctx) {
        p->input_frames = av_buffer_ref(input->hwctx);
        if (!p->input_frames)
            goto fail;
        p->ctx = device->cuda_ctx;
    }
    CUcontext dummy;
    if (!CUDA_CHECK(f, p->cu->cuCtxPushCurrent(p->ctx)))
        goto fail;

    struct mp_image *output = NULL;
    AVFrame *avframe = NULL;
    if (!p->configured && !configure(f, input, out_w, out_h, matrix, range))
        goto cuda_fail;
    // NVDEC may still be copying its planes. Order our reads after that work.
    if (!CUDA_CHECK(f, p->cu->cuEventRecord(p->decoded, device->stream)) ||
        !CUDA_CHECK(f, p->cu->cuStreamWaitEvent(p->stream, p->decoded, 0)) ||
        !yuv_to_rgba_convert(&p->converter, p->cu, f->log, input,
                             p->vsr.input.pixels, p->vsr.input.pitch, p->stream) ||
        !vsr_process(&p->vsr, &p->runtime, f->log))
        goto cuda_fail;

    avframe = av_frame_alloc();
    if (!avframe || av_hwframe_get_buffer(p->output_frames, avframe, 0) < 0) {
        MP_ERR(f, "VSR: cannot allocate CUDA/RGBA output frame.\n");
        goto cuda_fail;
    }
    CUDA_MEMCPY2D copy = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = (CUdeviceptr)p->vsr.output.pixels,
        .srcPitch = p->vsr.output.pitch,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstDevice = (CUdeviceptr)avframe->data[0],
        .dstPitch = avframe->linesize[0],
        .WidthInBytes = (size_t)out_w * 4,
        .Height = out_h,
    };
    if (!CUDA_CHECK(f, p->cu->cuMemcpy2DAsync(&copy, p->stream)) ||
        !CUDA_CHECK(f, p->cu->cuStreamSynchronize(p->stream)))
        goto cuda_fail;
    // Finished GPU work before freeing the input or handing output to the VO.
    // Each output owns its buffer; subsequent Run calls cannot overwrite it.
    output = mp_image_from_av_frame(avframe);
    av_frame_free(&avframe);
    if (!output)
        goto cuda_fail;
    mp_image_copy_attributes(output, input);
    output->params.repr = (struct pl_color_repr){
        .sys = PL_COLOR_SYSTEM_RGB,
        .levels = PL_COLOR_LEVELS_FULL,
        .alpha = PL_ALPHA_INDEPENDENT,
    };
    output->params.sys_orig = PL_COLOR_SYSTEM_RGB;
    output->params.chroma_location = PL_CHROMA_UNKNOWN;
    struct mp_rect crop = input->params.crop;
    output->params.crop = (struct mp_rect){
        lrint(crop.x0 * (double)out_w / input->w),
        lrint(crop.y0 * (double)out_h / input->h),
        lrint(crop.x1 * (double)out_w / input->w),
        lrint(crop.y1 * (double)out_h / input->h),
    };
    // Rounding noninteger output dimensions must not change display aspect.
    AVRational par = {input->params.p_w, input->params.p_h};
    if (par.num <= 0 || par.den <= 0)
        par = (AVRational){1, 1};
    par = av_mul_q(par, av_div_q((AVRational){input->w, input->h},
                                 (AVRational){out_w, out_h}));
    output->params.p_w = par.num;
    output->params.p_h = par.den;
    if (!CUDA_CHECK(f, p->cu->cuCtxPopCurrent(&dummy))) {
        talloc_free(output);
        goto fail;
    }
    mp_frame_unref(&frame);
    mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_VIDEO, output));
    return;

cuda_fail:
    // Preserve input/output memory until even partially submitted work ends.
    if (p->stream)
        CUDA_CHECK(f, p->cu->cuStreamSynchronize(p->stream));
    av_frame_free(&avframe);
    CUDA_CHECK(f, p->cu->cuCtxPopCurrent(&dummy));
fail:
    MP_ERR(f, "VSR: frame processing failed.\n");
    mp_frame_unref(&frame);
    mp_filter_internal_mark_failed(f);
}

static void reset(struct mp_filter *f)
{
    // NvVFX may carry temporal history; seeking starts a fresh effect.
    release_pipeline(f);
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    release_pipeline(f);
    vsr_runtime_unload(&p->runtime);
    cuda_free_functions(&p->cu);
}

static const struct mp_filter_info filter = {
    .name = "vsr",
    .priv_size = sizeof(struct priv),
    .process = process,
    .reset = reset,
    .destroy = destroy,
};

static struct mp_filter *create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }
    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    const double scales[] = {4.0 / 3, 1.5, 2, 3, 4};
    bool valid = false;
    for (int i = 0; i < MP_ARRAY_SIZE(scales); i++) {
        if (fabs(p->opts->scale - scales[i]) < 0.000001) {
            p->opts->scale = scales[i];
            valid = true;
            break;
        }
    }
    if (!valid) {
        MP_ERR(f, "VSR: scale must be 4/3 (1.333333), 1.5, 2, 3 or 4.\n");
        goto fail;
    }
    // Check NvVFX before touching CUDA: missing runtimes fail without GPU work.
    if (!vsr_runtime_load(&p->runtime, f->log))
        goto fail;
    if (cuda_load_functions(&p->cu, NULL) < 0) {
        MP_ERR(f, "VSR: NVIDIA CUDA driver unavailable.\n");
        goto fail;
    }
    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");
    return f;
fail:
    talloc_free(f);
    return NULL;
}

#define OPT_BASE_STRUCT struct vsr_opts
static const m_option_t options[] = {
    {"scale", OPT_DOUBLE(scale), M_RANGE(1.333332, 4)},
    {"quality", OPT_CHOICE(quality, {"low", 1}, {"medium", 2},
                           {"high", 3}, {"ultra", 4},
                           {"highbitrate-low", 16}, {"highbitrate-medium", 17},
                           {"highbitrate-high", 18}, {"highbitrate-ultra", 19})},
    {0}
};

const struct mp_user_filter_entry vf_vsr = {
    .desc = {
        .name = "vsr",
        .description = "NVIDIA NvVFX fixed-scale CUDA super resolution",
        .priv_size = sizeof(struct vsr_opts),
        .priv_defaults = &(const struct vsr_opts){.scale = 2, .quality = 3},
        .options = options,
    },
    .create = create,
};
