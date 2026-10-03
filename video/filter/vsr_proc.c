/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#include "vsr_internal.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "mpv_talloc.h"

#define NVVFX_DEFAULT_LIBDIR "/home/mathias/git/mpv-nvvfx/local/nvvfx/lib"

void vsr_runtime_unload(struct vsr_runtime *r)
{
    for (int i = r->num_libs - 1; i >= 0; i--)
        dlclose(r->libs[i]);
    talloc_free(r->dir);
    *r = (struct vsr_runtime){0};
}

static bool load_directory(struct vsr_runtime *r, struct mp_log *log,
                           const char *dir)
{
    // NVIDIA VFX 0.2.0.0: preload every dependency globally before consumers.
    // Absolute paths avoid relying on the wheel-specific RPATH entries.
    const char *libs[] = {
        "libcudart-256e6409.so.12.9.79",
        "libnppc.so.12", "libnppial.so.12", "libnppicc.so.12", "libnppidei.so.12",
        "libnppif.so.12", "libnppig.so.12", "libnppim.so.12", "libnppist.so.12",
        "libnppitc.so.12", "libcudnn.so.9", "libNVCVImage.so",
        "libnvngxruntime.so", "libVideoFXLocal.so", "libVideoFX.so",
        "libnvVFXVideoSuperRes.so", "libnvidia-ngx-vsr.so.1.8.3",
    };
    void *image_lib = NULL, *vfx_lib = NULL;
    for (int i = 0; i < MP_ARRAY_SIZE(libs); i++) {
        char *path = talloc_asprintf(NULL, "%s/%s", dir, libs[i]);
        void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            mp_warn(log, "VSR: cannot load %s (%s): %s\n",
                    libs[i], path, dlerror());
            talloc_free(path);
            goto fail;
        }
        talloc_free(path);
        r->libs[r->num_libs++] = handle;
        if (!strcmp(libs[i], "libNVCVImage.so"))
            image_lib = handle;
        if (!strcmp(libs[i], "libVideoFX.so"))
            vfx_lib = handle;
    }

#define SYM(field, lib, name) do { \
    *(void **)(&r->field) = dlsym(lib, name); \
    if (!r->field) { \
        mp_warn(log, "VSR: missing %s in %s: %s\n", name, dir, dlerror()); \
        goto fail; \
    } \
} while (0)
    SYM(create, vfx_lib, "NvVFX_CreateEffect");
    SYM(destroy, vfx_lib, "NvVFX_DestroyEffect");
    SYM(set_u32, vfx_lib, "NvVFX_SetU32");
    SYM(set_image, vfx_lib, "NvVFX_SetImage");
    SYM(set_stream, vfx_lib, "NvVFX_SetCudaStream");
    SYM(load, vfx_lib, "NvVFX_Load");
    SYM(run, vfx_lib, "NvVFX_Run");
    SYM(alloc, image_lib, "NvCVImage_Alloc");
    SYM(dealloc, image_lib, "NvCVImage_Dealloc");
#undef SYM
    r->dir = talloc_strdup(NULL, dir);
    return true;
fail:
    vsr_runtime_unload(r);
    return false;
}

bool vsr_runtime_load(struct vsr_runtime *r, struct mp_log *log)
{
    const char *override = getenv("MPV_NVVFX_LIBDIR");
    if (override && override[0]) {
        if (override[0] != '/') {
            mp_err(log, "VSR: MPV_NVVFX_LIBDIR must be an absolute path; "
                   "current-directory library lookup is disabled.\n");
            return false;
        }
        if (load_directory(r, log, override))
            return true;
    }
    if (load_directory(r, log, NVVFX_DEFAULT_LIBDIR))
        return true;
    mp_err(log, "VSR: NvVFX runtime unavailable. Set MPV_NVVFX_LIBDIR to an "
           "absolute directory containing libNVCVImage.so, libVideoFX.so "
           "and their dependencies, or provide them in %s.\n",
           NVVFX_DEFAULT_LIBDIR);
    return false;
}

static bool vfx_check(struct mp_log *log, int status, const char *operation)
{
    if (!status)
        return true;
    mp_err(log, "VSR: NvVFX %s failed (status %d). Check runtime, models "
           "and GPU compatibility.\n", operation, status);
    return false;
}

void vsr_destroy(struct vsr_context *c, struct vsr_runtime *r)
{
    if (c->effect)
        r->destroy(c->effect);
    if (c->input.pixels)
        r->dealloc(&c->input);
    if (c->output.pixels)
        r->dealloc(&c->output);
    *c = (struct vsr_context){0};
}

bool vsr_init(struct vsr_context *c, struct vsr_runtime *r,
              struct mp_log *log, int in_w, int in_h, int out_w, int out_h,
              int quality, CUstream stream)
{
    c->stream = stream;
    // NVCV_RGBA=6, NVCV_U8=1, chunky layout=0, GPU memory=1.
    if (!vfx_check(log, r->create("VideoSuperRes", &c->effect), "CreateEffect") ||
        !vfx_check(log, r->alloc(&c->input, in_w, in_h, 6, 1, 0, 1, 32), "Alloc(input)") ||
        !vfx_check(log, r->alloc(&c->output, out_w, out_h, 6, 1, 0, 1, 32), "Alloc(output)") ||
        !vfx_check(log, r->set_u32(c->effect, "QualityLevel", quality), "QualityLevel") ||
        !vfx_check(log, r->set_image(c->effect, "SrcImage0", &c->input), "SrcImage0") ||
        !vfx_check(log, r->set_image(c->effect, "DstImage0", &c->output), "DstImage0") ||
        !vfx_check(log, r->set_stream(c->effect, "CudaStream", stream), "CudaStream") ||
        !vfx_check(log, r->load(c->effect), "Load"))
        return false;
    mp_info(log, "VSR: VideoSuperRes %dx%d -> %dx%d, quality=%d\n",
            in_w, in_h, out_w, out_h, quality);
    return true;
}

bool vsr_process(struct vsr_context *c, struct vsr_runtime *r,
                 struct mp_log *log)
{
    return vfx_check(log, r->run(c->effect, 0), "Run");
}
