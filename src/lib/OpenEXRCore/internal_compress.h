/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#ifndef OPENEXR_CORE_COMPRESS_H
#define OPENEXR_CORE_COMPRESS_H

#include "openexr_encode.h"

uint64_t internal_rle_compress (
    void* out, uint64_t outbytes, const void* src, uint64_t srcbytes);

void internal_zip_deconstruct_bytes (
    uint8_t* scratch, const uint8_t* source, uint64_t count);

void internal_zip_reconstruct_bytes (
    uint8_t* out, uint8_t* scratch_source, uint64_t count);

exr_result_t internal_exr_apply_rle (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_zip (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_piz (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_pxr24 (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_b44 (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_b44a (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_dwaa (exr_encode_pipeline_t* encode);

exr_result_t internal_exr_apply_dwab (exr_encode_pipeline_t* encode);

#ifdef __cplusplus
extern "C" {
#endif
exr_result_t internal_exr_apply_ht (exr_encode_pipeline_t* encode);
#ifdef __cplusplus
}
#endif

exr_result_t internal_exr_apply_zstd (exr_encode_pipeline_t* encode);

/* Upper bound on the size of a zlib stream of in_bytes (an empty input
 * still produces a few bytes of header and footer). Unlike exr_compress_max_buffer_size it does not also reserve room
 * for the Huffman coder's working tables, so it is a bound on the output
 * only. */
size_t internal_exr_deflate_bound (size_t in_bytes);
#endif /* OPENEXR_CORE_COMPRESS_H */
