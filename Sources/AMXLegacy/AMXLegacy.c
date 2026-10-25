#include "AMXLegacy.h"

// See the header: the migration to `BNNSGraph` was measured and rejected.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

ssize_t amx_matmul_workspace(bool transA, bool transB, float alpha,
                             const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
                             const BNNSNDArrayDescriptor *C, const BNNSFilterParameters *params) {
    return BNNSMatMulWorkspaceSize(transA, transB, alpha, A, B, C, params);
}

int amx_matmul(bool transA, bool transB, float alpha,
               const BNNSNDArrayDescriptor *A, const BNNSNDArrayDescriptor *B,
               const BNNSNDArrayDescriptor *C, void *workspace, const BNNSFilterParameters *params) {
    return BNNSMatMul(transA, transB, alpha, A, B, C, workspace, params);
}

#pragma clang diagnostic pop

#include <arm_neon.h>

static inline float32x4_t exponential4(float32x4_t t) {
    float32x4_t x = vmaxq_f32(t, vdupq_n_f32(-87.3365f));
    x = vminq_f32(x, vdupq_n_f32(0.0f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.442695041f));
    // r = x - n * ln 2, ln 2 in two parts so that n * C1 is exact.
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmaq_f32(r, n, vdupq_n_f32(2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
    float32x4_t y = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), vmulq_f32(p, r), r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

float amx_softmax_exponentiate(float *row, long count, float scale, float offset) {
    float32x4_t low = vdupq_n_f32(0.0f), high = vdupq_n_f32(0.0f);
    float32x4_t vs = vdupq_n_f32(scale), vo = vdupq_n_f32(offset);
    long j = 0;
    for (; j + 8 <= count; j += 8) {
        float32x4_t ea = exponential4(vfmaq_f32(vo, vld1q_f32(row + j), vs));
        float32x4_t eb = exponential4(vfmaq_f32(vo, vld1q_f32(row + j + 4), vs));
        vst1q_f32(row + j, ea); vst1q_f32(row + j + 4, eb);
        low = vaddq_f32(low, ea); high = vaddq_f32(high, eb);
    }
    float tail = 0.0f;
    for (; j < count; j++) {
        float e = vgetq_lane_f32(exponential4(vdupq_n_f32(fmaf(row[j], scale, offset))), 0);
        row[j] = e; tail += e;
    }
    return vaddvq_f32(vaddq_f32(low, high)) + tail;
}
