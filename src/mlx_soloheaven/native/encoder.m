// dsv4 native decode runtime — the C encode loop (Stage 3b).
//
// Python-side ctypes re-encoding measured 11 us/dispatch of FFI overhead
// (docs/benchmarks/deepseek-v4.md, ladder step 1); this loop exists to make
// per-token re-encoding of a ~1500-dispatch plan cost ~1 ms. Per-token
// VARYING scalars never touch the plan: they live in a small uniform
// MTLBuffer whose contents Python rewrites before each commit (unified
// memory). setBytes here is only for plan-static constants (e.g. qmv K/N).
//
// Build: clang -fobjc-arc -O2 -shared -framework Metal -framework Foundation
//        encoder.m -o libdsv4enc.dylib   (runtime.py does this on demand)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>

#define DSV4_MAX_BUFS 16
#define DSV4_MAX_BYTES 4

typedef struct {
    int32_t pso;                       // index into pipeline table
    int32_t n_bufs;
    int32_t buf_ids[DSV4_MAX_BUFS];    // indices into session buffer table
    uint64_t buf_offs[DSV4_MAX_BUFS];
    int32_t buf_slots[DSV4_MAX_BUFS];  // [[buffer(slot)]] index
    int32_t n_bytes;
    int32_t bytes_off[DSV4_MAX_BYTES]; // offset into the constants blob
    int32_t bytes_len[DSV4_MAX_BYTES];
    int32_t bytes_slot[DSV4_MAX_BYTES];
    uint64_t grid[3];
    uint64_t group[3];
    int32_t barrier;                   // 1: buffer barrier BEFORE this dispatch
} DSV4PlanItem;

int dsv4_encode_commit(
    void* queue_v,
    const DSV4PlanItem* items,
    int n_items,
    void* const* psos_v,
    void* const* bufs_v,
    const uint8_t* const_blob,
    int wait)
{
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_v;
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        int32_t last_pso = -1;
        for (int k = 0; k < n_items; ++k) {
            const DSV4PlanItem* it = &items[k];
            // MLX buffers are hazard-untracked, so a dispatch that reads what
            // the previous one wrote needs an explicit buffer barrier.
            if (it->barrier) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            if (it->pso != last_pso) {
                [enc setComputePipelineState:
                     (__bridge id<MTLComputePipelineState>)psos_v[it->pso]];
                last_pso = it->pso;
            }
            for (int i = 0; i < it->n_bufs; ++i) {
                [enc setBuffer:(__bridge id<MTLBuffer>)bufs_v[it->buf_ids[i]]
                        offset:it->buf_offs[i]
                       atIndex:(NSUInteger)it->buf_slots[i]];
            }
            for (int i = 0; i < it->n_bytes; ++i) {
                [enc setBytes:const_blob + it->bytes_off[i]
                       length:(NSUInteger)it->bytes_len[i]
                      atIndex:(NSUInteger)it->bytes_slot[i]];
            }
            MTLSize grid = MTLSizeMake(it->grid[0], it->grid[1], it->grid[2]);
            MTLSize group = MTLSizeMake(it->group[0], it->group[1], it->group[2]);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
        }
        [enc endEncoding];
        [cmd commit];
        if (wait) {
            [cmd waitUntilCompleted];
            if (cmd.status == MTLCommandBufferStatusError) return -1;
        }
        return 0;
    }
}

// Utility used by runtime.py during buffer-table construction: contents
// pointer of an MLX-owned buffer (unified memory) so Python can write the
// per-token uniform values without any Metal call.
void* dsv4_buffer_contents(void* buf_v) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_v;
    return buf.contents;
}

uint64_t dsv4_buffer_length(void* buf_v) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_v;
    return buf.length;
}
