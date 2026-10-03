/*
 * Derived from zhangmq/vsr-player, commit
 * f7941984e29c687bf2e318646728f06ee5e5cd0a (same relative filename).
 * Modified for mpv's fixed-scale NVDEC VSR filter; see DOCS/nvvfx-vsr.md.
 * SPDX-License-Identifier: GPL-2.0-only
 * Distributed under the GNU GPL version 2; see LICENSE.GPL.
 */

#include "yuv_to_rgba.h"

#include <dlfcn.h>
#include <nvrtc.h>
#include <stdlib.h>
#include <stdio.h>

#include "common/msg.h"

static const char *kKernelSrc =
"extern \"C\" __global__ void yuv_to_rgba_u8(\n"
"    const unsigned char* __restrict__ y_plane,\n"
"    int y_pitch,\n"
"    const unsigned char* __restrict__ uv_plane,\n"
"    int uv_pitch,\n"
"    unsigned char* __restrict__ rgba_output,\n"
"    int rgba_pitch,\n"
"    int width,\n"
"    int height)\n"
"{\n"
"    int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    float Y, U, V;\n"
"    if (BIT_DEPTH <= 8) {\n"
"        Y = (float)y_plane[(unsigned long long)y * y_pitch + x];\n"
"    } else {\n"
"        const unsigned short* y16 = (const unsigned short*)y_plane;\n"
"        Y = (float)y16[(unsigned long long)y * y_pitch / 2 + x];\n"
"    }\n"
"\n"
"    int uv_x = x & ~1;\n"
"    int uv_y = y / 2;\n"
"    if (BIT_DEPTH <= 8) {\n"
"        U = (float)uv_plane[(unsigned long long)uv_y * uv_pitch + uv_x];\n"
"        V = (float)uv_plane[(unsigned long long)uv_y * uv_pitch + uv_x + 1];\n"
"    } else {\n"
"        const unsigned short* uv16 = (const unsigned short*)uv_plane;\n"
"        unsigned long long idx = (unsigned long long)uv_y * (uv_pitch / 2) + uv_x;\n"
"        U = (float)uv16[idx];\n"
"        V = (float)uv16[idx + 1];\n"
"    }\n"
"\n"
"    float unit = (BIT_DEPTH <= 8) ? 1.0f : 256.0f;\n"
"    float maximum = (BIT_DEPTH <= 8) ? 255.0f : 65535.0f;\n"
"    if (YUV_RANGE_LIMITED) {\n"
"        Y = (Y - 16.0f * unit) / (219.0f * unit);\n"
"        U = (U - 128.0f * unit) / (224.0f * unit);\n"
"        V = (V - 128.0f * unit) / (224.0f * unit);\n"
"    } else {\n"
"        Y /= maximum;\n"
"        U = (U - 128.0f * unit) / maximum;\n"
"        V = (V - 128.0f * unit) / maximum;\n"
"    }\n"
"\n"
"    float r = __saturatef(Y + K_RV * V);\n"
"    float g = __saturatef(Y - K_GU * U - K_GV * V);\n"
"    float b = __saturatef(Y + K_BU * U);\n"
"\n"
"    unsigned long long out_idx = (unsigned long long)y * rgba_pitch + x * 4;\n"
"    rgba_output[out_idx + 0] = (unsigned char)(r * 255.0f + 0.5f);\n"
"    rgba_output[out_idx + 1] = (unsigned char)(g * 255.0f + 0.5f);\n"
"    rgba_output[out_idx + 2] = (unsigned char)(b * 255.0f + 0.5f);\n"
"    rgba_output[out_idx + 3] = 255;\n"
"}\n";

// ── Matrix coefficient helpers ─────────────────────────────────────────

static void get_matrix_defines(enum yuv_matrix m,
                                const char **k_rv, const char **k_gu,
                                const char **k_gv, const char **k_bu)
{
    // Kr + Kg + Kb = 1.0 for all matrices, coefficients pre-computed:
    // K_RV = 2*(1-Kr), K_GU = 2*(1-Kb)*Kb/Kg, K_GV = 2*(1-Kr)*Kr/Kg, K_BU = 2*(1-Kb)
    static const char *rv_601 = "1.402f",  *gu_601 = "0.34414f";
    static const char *gv_601 = "0.71414f", *bu_601 = "1.772f";
    static const char *rv_709 = "1.5748f",  *gu_709 = "0.18732f";
    static const char *gv_709 = "0.46812f", *bu_709 = "1.8556f";
    static const char *rv_2020 = "1.4746f",  *gu_2020 = "0.16455f";
    static const char *gv_2020 = "0.57135f", *bu_2020 = "1.8814f";

    switch (m) {
    case YUV_MATRIX_BT709:
        *k_rv = rv_709; *k_gu = gu_709; *k_gv = gv_709; *k_bu = bu_709; break;
    case YUV_MATRIX_BT2020:
        *k_rv = rv_2020; *k_gu = gu_2020; *k_gv = gv_2020; *k_bu = bu_2020; break;
    default:
        *k_rv = rv_601; *k_gu = gu_601; *k_gv = gv_601; *k_bu = bu_601; break;
    }
}

// NVRTC is optional at mpv startup, just like NvVFX. Prefer the runtime bundle,
// then the system loader's versioned toolkit library (no toolkit rpath).
bool yuv_to_rgba_init(struct yuv_to_rgba *c, CudaFunctions *cu,
                      struct mp_log *log, const char *runtime_dir,
                      int bit_depth, enum yuv_matrix matrix, enum yuv_range range)
{
    // Its SONAME matches NVRTC's internal dlopen name, so preload by absolute
    // path and keep it visible/alive until NVRTC has finished using it.
    const char *builtins_name = "libnvrtc-builtins.so.13.2";
    char *path = talloc_asprintf(NULL, "%s/%s", runtime_dir, builtins_name);
    void *builtins = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!builtins) {
        mp_err(log, "VSR: cannot preload %s (%s): %s\n",
               builtins_name, path, dlerror());
        talloc_free(path);
        return false;
    }
    talloc_free(path);
    path = talloc_asprintf(NULL, "%s/libnvrtc.so.13", runtime_dir);
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    talloc_free(path);
    if (!lib)
        lib = dlopen("libnvrtc.so.13", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        mp_err(log, "VSR: NVRTC unavailable (libnvrtc.so.13): %s\n", dlerror());
        dlclose(builtins);
        return false;
    }

#define NVRTC_FUNCTIONS(F) \
    F(nvrtcCreateProgram) F(nvrtcCompileProgram) F(nvrtcGetProgramLogSize) \
    F(nvrtcGetProgramLog) F(nvrtcGetPTXSize) F(nvrtcGetPTX) \
    F(nvrtcDestroyProgram) F(nvrtcGetErrorString)
#define DECLARE(name) __typeof__(&name) p_##name = NULL;
    NVRTC_FUNCTIONS(DECLARE)
#undef DECLARE
    nvrtcProgram prog = NULL;
    char *ptx = NULL;
    bool ok = false;
#define LOAD(name) do { \
    *(void **)(&p_##name) = dlsym(lib, #name); \
    if (!p_##name) { \
        mp_err(log, "VSR: missing NVRTC symbol %s: %s\n", #name, dlerror()); \
        goto done; \
    } \
} while (0);
    NVRTC_FUNCTIONS(LOAD)
#undef LOAD
#undef NVRTC_FUNCTIONS

    CUdevice dev;
    int major, minor;
    if (!vsr_cuda_check(log, cu, cu->cuCtxGetDevice(&dev), "cuCtxGetDevice") ||
        !vsr_cuda_check(log, cu, cu->cuDeviceComputeCapability(&major, &minor, dev),
                        "cuDeviceComputeCapability"))
        goto done;

    const char *rv, *gu, *gv, *bu;
    get_matrix_defines(matrix, &rv, &gu, &gv, &bu);
    char arch[64], bd[32], rn[32], krv[64], kgu[64], kgv[64], kbu[64];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
    snprintf(bd, sizeof(bd), "-DBIT_DEPTH=%d", bit_depth);
    snprintf(rn, sizeof(rn), "-DYUV_RANGE_LIMITED=%d", range == YUV_RANGE_LIMITED);
    snprintf(krv, sizeof(krv), "-DK_RV=%s", rv);
    snprintf(kgu, sizeof(kgu), "-DK_GU=%s", gu);
    snprintf(kgv, sizeof(kgv), "-DK_GV=%s", gv);
    snprintf(kbu, sizeof(kbu), "-DK_BU=%s", bu);
    const char *options[] = {arch, "--use_fast_math", bd, rn, krv, kgu, kgv, kbu};

    nvrtcResult result = p_nvrtcCreateProgram(&prog, kKernelSrc, "vsr_yuv.cu",
                                              0, NULL, NULL);
    if (result != NVRTC_SUCCESS) {
        mp_err(log, "VSR: nvrtcCreateProgram: %s\n", p_nvrtcGetErrorString(result));
        goto done;
    }
    result = p_nvrtcCompileProgram(prog, MP_ARRAY_SIZE(options), options);
    if (result != NVRTC_SUCCESS) {
        size_t size = 0;
        p_nvrtcGetProgramLogSize(prog, &size);
        char *message = talloc_zero_size(NULL, size + 1);
        p_nvrtcGetProgramLog(prog, message);
        mp_err(log, "VSR: NVRTC compile: %s\n%s\n",
               p_nvrtcGetErrorString(result), message);
        talloc_free(message);
        goto done;
    }
    size_t size = 0;
    result = p_nvrtcGetPTXSize(prog, &size);
    if (result == NVRTC_SUCCESS) {
        ptx = talloc_size(NULL, size);
        result = p_nvrtcGetPTX(prog, ptx);
    }
    if (result != NVRTC_SUCCESS) {
        mp_err(log, "VSR: NVRTC PTX retrieval: %s\n", p_nvrtcGetErrorString(result));
        goto done;
    }
    if (!vsr_cuda_check(log, cu, cu->cuModuleLoadData(&c->module, ptx), "cuModuleLoadData") ||
        !vsr_cuda_check(log, cu, cu->cuModuleGetFunction(&c->kernel, c->module,
                         "yuv_to_rgba_u8"), "cuModuleGetFunction"))
        goto done;
    ok = true;
done:
    if (prog)
        p_nvrtcDestroyProgram(&prog);
    talloc_free(ptx);
    dlclose(lib);
    dlclose(builtins);
    if (!ok)
        yuv_to_rgba_destroy(c, cu);
    return ok;
}

bool yuv_to_rgba_convert(struct yuv_to_rgba *c, CudaFunctions *cu,
                         struct mp_log *log, struct mp_image *input,
                         void *output, int output_pitch, CUstream stream)
{
    void *args[] = {
        &input->planes[0], &input->stride[0],
        &input->planes[1], &input->stride[1],
        &output, &output_pitch, &input->w, &input->h,
    };
    return vsr_cuda_check(log, cu, cu->cuLaunchKernel(c->kernel,
        (input->w + 15) / 16, (input->h + 15) / 16, 1,
        16, 16, 1, 0, stream, args, NULL), "cuLaunchKernel");
}

void yuv_to_rgba_destroy(struct yuv_to_rgba *c, CudaFunctions *cu)
{
    if (c->module)
        cu->cuModuleUnload(c->module);
    *c = (struct yuv_to_rgba){0};
}
