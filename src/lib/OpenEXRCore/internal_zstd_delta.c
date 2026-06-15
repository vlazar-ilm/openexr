#include <stdint.h>
#include <string.h>

/* All delta kernels are out-of-place: (uint8_t* dst, const uint8_t* src, n).
 * Calling with dst == src reproduces the previous in-place behaviour, so every
 * call site is free to either keep working in place or fold the delta pass into
 * a buffer move. The float block can begin at a 2-byte-but-not-4-byte offset, so
 * all element access goes through these unaligned-safe memcpy helpers rather than
 * raw aligned pointer casts. */

static inline uint16_t
ld16 (const uint8_t* p)
{
    uint16_t v;
    memcpy (&v, p, 2);
    return v;
}
static inline void
st16 (uint8_t* p, uint16_t v)
{
    memcpy (p, &v, 2);
}
static inline uint32_t
ld32 (const uint8_t* p)
{
    uint32_t v;
    memcpy (&v, p, 4);
    return v;
}
static inline void
st32 (uint8_t* p, uint32_t v)
{
    memcpy (p, &v, 4);
}

/* ========================================================================= */
/* SCALAR IMPLEMENTATIONS                                                    */
/* ========================================================================= */

static void
delta_encode_row_u16_scalar (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    /* Iterate top-down so the kernel is safe in place (dst == src): each store
     * at k reads only src[k] and src[k-1] (indices <= k, not yet overwritten).
     * For dst != src the order is irrelevant and the result is identical. */
    for (uint64_t k = n - 1; k > 0; --k)
        st16 (
            dst + k * 2,
            (uint16_t) ((unsigned) ld16 (src + k * 2) -
                        (unsigned) ld16 (src + (k - 1) * 2)));
    st16 (dst, ld16 (src)); /* base sample copied verbatim */
}

static void
delta_decode_row_u16_scalar (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    st16 (dst, ld16 (src));
    for (uint64_t k = 1; k < n; ++k)
        st16 (
            dst + k * 2,
            (uint16_t) ((unsigned) ld16 (src + k * 2) +
                        (unsigned) ld16 (dst + (k - 1) * 2)));
}

static void
delta_encode_row_u32_scalar (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    /* Top-down for in-place safety (see delta_encode_row_u16_scalar). */
    for (uint64_t k = n - 1; k > 0; --k)
        st32 (dst + k * 4, ld32 (src + k * 4) - ld32 (src + (k - 1) * 4));
    st32 (dst, ld32 (src)); /* base sample copied verbatim */
}

static void
delta_decode_row_u32_scalar (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    st32 (dst, ld32 (src));
    for (uint64_t k = 1; k < n; ++k)
        st32 (dst + k * 4, ld32 (src + k * 4) + ld32 (dst + (k - 1) * 4));
}

/* ========================================================================= */
/* AVX2 IMPLEMENTATIONS                                                      */
/* ========================================================================= */

#if defined(__GNUC__) && defined(__x86_64__)
#    include <immintrin.h>

/* Inactive: AVX2 encode kernels, kept ready to enable. Out-of-place
 * (dst, src) and top-down like the scalar encode, so they stay correct when
 * called in place (dst == src). To enable, uncomment these and the matching
 * branches in delta_encode_row_u16 / delta_encode_row_u32 below. */
/*
__attribute__ ((target ("avx2"))) static void
delta_encode_row_u16_avx2 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    uint64_t i = n;
    for (; i >= 16 + 1; i -= 16)
    {
        __m256i curr =
            _mm256_loadu_si256 ((const __m256i*) (src + (i - 16) * 2));
        __m256i prev =
            _mm256_loadu_si256 ((const __m256i*) (src + (i - 17) * 2));
        __m256i delta = _mm256_sub_epi16 (curr, prev);
        _mm256_storeu_si256 ((__m256i*) (dst + (i - 16) * 2), delta);
    }
    for (uint64_t k = i - 1; k > 0; --k)
        st16 (
            dst + k * 2,
            (uint16_t) ((unsigned) ld16 (src + k * 2) -
                        (unsigned) ld16 (src + (k - 1) * 2)));
    st16 (dst, ld16 (src));
}

__attribute__ ((target ("avx2"))) static void
delta_encode_row_u32_avx2 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    uint64_t i = n;
    for (; i >= 8 + 1; i -= 8)
    {
        __m256i curr =
            _mm256_loadu_si256 ((const __m256i*) (src + (i - 8) * 4));
        __m256i prev =
            _mm256_loadu_si256 ((const __m256i*) (src + (i - 9) * 4));
        __m256i delta = _mm256_sub_epi32 (curr, prev);
        _mm256_storeu_si256 ((__m256i*) (dst + (i - 8) * 4), delta);
    }
    for (uint64_t k = i - 1; k > 0; --k)
        st32 (dst + k * 4, ld32 (src + k * 4) - ld32 (src + (k - 1) * 4));
    st32 (dst, ld32 (src));
}
*/

__attribute__ ((target ("avx2"))) static void
delta_decode_row_u16_avx2 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    uint64_t i = 1;
    uint16_t prev_sum;
    prev_sum = ld16 (src);
    st16 (dst, prev_sum);

    __m256i bcast_mask = _mm256_set_epi8 (
        15,
        14,
        15,
        14,
        15,
        14,
        15,
        14,
        15,
        14,
        15,
        14,
        15,
        14,
        15,
        14,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1);

    for (; i + 16 <= n; i += 16)
    {
        __m256i x = _mm256_loadu_si256 ((const __m256i*) (src + i * 2));
        x         = _mm256_add_epi16 (x, _mm256_slli_si256 (x, 2));
        x         = _mm256_add_epi16 (x, _mm256_slli_si256 (x, 4));
        x         = _mm256_add_epi16 (x, _mm256_slli_si256 (x, 8));

        __m256i lane_shift = _mm256_permute2x128_si256 (x, x, 0x08);
        lane_shift         = _mm256_shuffle_epi8 (lane_shift, bcast_mask);
        x                  = _mm256_add_epi16 (x, lane_shift);

        x = _mm256_add_epi16 (x, _mm256_set1_epi16 (prev_sum));
        _mm256_storeu_si256 ((__m256i*) (dst + i * 2), x);

        __m128i hi_lane = _mm256_extracti128_si256 (x, 1);
        prev_sum        = (uint16_t) _mm_extract_epi16 (hi_lane, 7);
    }

    for (uint64_t k = i; k < n; ++k)
        st16 (
            dst + k * 2,
            (uint16_t) ((unsigned) ld16 (src + k * 2) +
                        (unsigned) ld16 (dst + (k - 1) * 2)));
}

__attribute__ ((target ("avx2"))) static void
delta_decode_row_u32_avx2 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    if (n == 0) return;
    uint64_t i = 1;
    uint32_t prev_sum;
    prev_sum = ld32 (src);
    st32 (dst, prev_sum);

    for (; i + 8 <= n; i += 8)
    {
        __m256i x = _mm256_loadu_si256 ((const __m256i*) (src + i * 4));
        x         = _mm256_add_epi32 (x, _mm256_slli_si256 (x, 4));
        x         = _mm256_add_epi32 (x, _mm256_slli_si256 (x, 8));

        __m256i lane_shift = _mm256_permute2x128_si256 (x, x, 0x08);
        lane_shift =
            _mm256_shuffle_epi32 (lane_shift, _MM_SHUFFLE (3, 3, 3, 3));
        x = _mm256_add_epi32 (x, lane_shift);

        x = _mm256_add_epi32 (x, _mm256_set1_epi32 (prev_sum));
        _mm256_storeu_si256 ((__m256i*) (dst + i * 4), x);

        __m128i hi_lane = _mm256_extracti128_si256 (x, 1);
        prev_sum        = (uint32_t) _mm_extract_epi32 (hi_lane, 3);
    }

    for (uint64_t k = i; k < n; ++k)
        st32 (dst + k * 4, ld32 (src + k * 4) + ld32 (dst + (k - 1) * 4));
}

#endif

/* ========================================================================= */
/* CLIENT-FACING DISPATCH API                                                */
/* ========================================================================= */

void
delta_encode_row_u16 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    /*#if defined(__GNUC__) && defined(__x86_64__)
    __builtin_cpu_init ();
    if (__builtin_cpu_supports ("avx2"))
    {
        delta_encode_row_u16_avx2 (dst, src, n);
        return;
    }
#endif*/
    delta_encode_row_u16_scalar (dst, src, n);
}

void
delta_decode_row_u16 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
#if defined(__GNUC__) && defined(__x86_64__)
    __builtin_cpu_init ();
    if (__builtin_cpu_supports ("avx2"))
    {
        delta_decode_row_u16_avx2 (dst, src, n);
        return;
    }
#endif
    delta_decode_row_u16_scalar (dst, src, n);
}

void
delta_encode_row_u32 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
    /*#if defined(__GNUC__) && defined(__x86_64__)
    __builtin_cpu_init ();
    if (__builtin_cpu_supports ("avx2"))
    {
        delta_encode_row_u32_avx2 (dst, src, n);
        return;
    }
#endif*/
    delta_encode_row_u32_scalar (dst, src, n);
}

void
delta_decode_row_u32 (uint8_t* dst, const uint8_t* src, uint64_t n)
{
#if defined(__GNUC__) && defined(__x86_64__)
    __builtin_cpu_init ();
    if (__builtin_cpu_supports ("avx2"))
    {
        delta_decode_row_u32_avx2 (dst, src, n);
        return;
    }
#endif
    delta_decode_row_u32_scalar (dst, src, n);
}
