// Exact K16/D512 convolutions of eight 16-bit key limbs. The host derives
// the reconstruction span from the prime so span*32*65535 < p/2. Each
// batch reconstructs into Fp128 before its results are accumulated further.
namespace akita {
struct OneHotLimbSource {
    ulong fields, chunks, hot_offset;
    uint first_ring, pad;
};
struct OneHotLimbParams {
    NttPrime prime;
    uint positions, columns, digits, rows, rings, task_groups, reconstruction_positions;
    uint first_chunk, parts, direct_output;
};

template <uint C, uint DOT_BATCH, uint MODULUS, typename KeyIndex>
kernel void akita_onehot_limb_commit(
    device const uchar* hot [[buffer(0)]],
    device const uint* key [[buffer(1)]],
    device const uint* tables [[buffer(2)]],
    device const OneHotLimbSource* sources [[buffer(3)]],
    device const uint* source_for_ring [[buffer(4)]],
    device jolt::Fp128<C>* out [[buffer(5)]],
    constant OneHotLimbParams& p [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    static_assert(DOT_BATCH == 4, "two four-position sources fill the shared tile");
    using F = jolt::Fp128<C>;
    uint pair = group % p.task_groups;
    uint part = group / p.task_groups;
    uint row_a = pair % p.rows;
    uint task[2] = {2 * (pair / p.rows) * p.rows + row_a, 0};
    task[1] = task[0] + p.rows;
    OneHotLimbSource src[2];
    src[0] = sources[source_for_ring[task[0]]];
    src[1] = src[0];
    if (task[1] < p.rings) src[1] = sources[source_for_ring[task[1]]];
    uint begin = (p.first_chunk + part) * p.reconstruction_positions;
    uint end = metal::min(p.positions, begin + p.reconstruction_positions);
    uint modulus = MODULUS;
    NttPrime prime = p.prime;
    prime.p = int(MODULUS);
    ulong live_chunks[2], source_base[2], block_chunks[2];
    for (uint s = 0; s < 2; s++) {
        uint block = (task[s] - src[s].first_ring) / p.rows;
        live_chunks[s] = task[s] < p.rings ? metal::min(src[s].chunks, src[s].fields / 16) : 0;
        block_chunks[s] = ulong(block) * p.positions * 32;
        source_base[s] = src[s].hot_offset + block_chunks[s];
    }
    if (block_chunks[0] + ulong(begin) * 32 >= live_chunks[0] &&
        block_chunks[1] + ulong(begin) * 32 >= live_chunks[1]) {
        for (uint s = 0; s < 2; s++) if (task[s] < p.rings)
            for (uint coeff = tid; coeff < 512; coeff += 256) {
                if (!p.direct_output || p.first_chunk == 0)
                    out[(ulong(part) * p.rings + task[s]) * 512 + coeff] = F::zero();
            }
        return;
    }
    KeyIndex key_row = KeyIndex(row_a) * p.columns * 4096;
    KeyIndex key_stride = KeyIndex(p.digits) * 4096;
    device const uint* tw = tables + 16384;
    device const uint* twq = tw + 8;
    device const uint* invw = twq + 8;
    device const uint* invwq = invw + 256;
    device const uint* untwist = invwq + 256;
    device const uint* untwistq = untwist + 512;
    threadgroup uint work[4096];
    uint2 acc[16] = {};
    for (uint first = begin; first < end; first += DOT_BATCH) {
        KeyIndex first_key = key_row + KeyIndex(first) * key_stride;
        if (tid < 2 * DOT_BATCH * 32) {
            uint s = tid / (DOT_BATCH * 32), r = (tid / 32) % DOT_BATCH;
            uint c = tid % 32, pos = first + r;
            for (uint v = 0; v < 16; v++) work[(s * DOT_BATCH + r) * 512 + v * 32 + c] = 0;
            ulong chunk = block_chunks[s] + ulong(pos) * 32;
            if (pos < end && chunk < live_chunks[s]) {
                uint present = uint(metal::min(32ul, live_chunks[s] - chunk));
                for (uint row = 0; row < present; row++) {
                    uint v = hot[source_base[s] + ulong(pos) * 32 + row];
                    if (v != 255u) {
                        uint i = (s * DOT_BATCH + r) * 512 + v * 32 + c;
                        work[i] = uint(csubp(prime, int(work[i] + tables[(v * 32 + row) * 32 + c])));
                    }
                }
                uint a[16];
                for (uint v = 0; v < 16; v++) a[v] = work[(s * DOT_BATCH + r) * 512 + v * 32 + c];
                for (uint half_len = 8; half_len >= 1; half_len /= 2)
                    for (uint start = 0; start < 16; start += 2 * half_len)
                        for (uint j = 0; j < half_len; j++) {
                            uint x = a[start + j], y = a[start + j + half_len];
                            a[start + j] = uint(csubp(prime, int(x + y)));
                            uint diff = uint(caddp(prime, int(x) - int(y)));
                            a[start + j + half_len] = j == 0 ? diff :
                                shoup_mul_canonical(diff, tw[j * (8 / half_len)], twq[j * (8 / half_len)], modulus);
                        }
                for (uint v = 0; v < 16; v++) {
                    uint rev = ((v & 1) << 3) | ((v & 2) << 1) | ((v & 4) >> 1) | ((v & 8) >> 3);
                    work[(s * DOT_BATCH + r) * 512 + rev * 32 + c] = a[v];
                }
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint idx = tid + i * 256, coeff = idx % 512;
            ulong2 sum = ulong2(0);
            for (uint r = 0; r < DOT_BATCH; r++) if (first + r < end) {
                uint k = key[first_key + KeyIndex(r) * key_stride + idx];
                uint2 values(work[r * 512 + coeff], work[(DOT_BATCH + r) * 512 + coeff]);
                sum += ulong(k) * ulong2(values);
            }
            // Four canonical products give S < 2^32*p, so the unsigned
            // Montgomery quotient is below 2p and needs one subtraction.
            uint2 correction = uint2(sum) * (0u - uint(p.prime.pinv));
            uint2 v = uint2((sum + ulong2(correction) * ulong(modulus)) >> 32);
            v = metal::select(v, v - modulus, v >= modulus);
            acc[i] += v;
            acc[i] = metal::select(acc[i], acc[i] - modulus, acc[i] >= modulus);
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    for (uint s = 0; s < 2; s++) {
        if (task[s] >= p.rings) break;
        for (uint i = 0; i < 16; i++) {
            uint idx = tid + i * 256;
            work[(idx / 512) * 512 + (metal::reverse_bits(idx % 512) >> 23)] = s == 0 ? acc[i].x : acc[i].y;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint len = 2; len <= 512; len *= 2) {
            uint half_len = len / 2;
            for (uint i = tid; i < 2048; i += 256) {
                uint limb = i / 256, b = i % 256, j = b % half_len, start = (b / half_len) * len;
                uint lo = limb * 512 + start + j, hi = lo + half_len;
                uint u = work[lo];
                uint v = shoup_mul_canonical(work[hi], invw[j * (512 / len)], invwq[j * (512 / len)], modulus);
                work[lo] = uint(csubp(prime, int(u + v)));
                work[hi] = uint(caddp(prime, int(u) - int(v)));
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
        for (uint coeff = tid; coeff < 512; coeff += 256) {
            long carry = 0; uint4 lo(0u);
            for (uint limb = 0; limb < 8; limb++) {
                uint residue = shoup_mul_canonical(work[limb * 512 + coeff], untwist[coeff], untwistq[coeff], modulus);
                long centered = residue > modulus / 2 ? long(residue) - modulus : long(residue);
                long v = centered + carry;
                lo[limb / 2] |= uint(ushort(v)) << (16 * (limb % 2));
                carry = v >> 16;
            }
            F value{jolt::fp128_detail::fold2_canonicalize<C>(lo, 0)};
            value = value + F::from_i64(carry * long(C));
            ulong index = (ulong(part) * p.rings + task[s]) * 512 + coeff;
            out[index] = p.direct_output && p.first_chunk != 0 ? out[index] + value : value;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
}

template <uint C>
kernel void akita_onehot_limb_combine(
    device const jolt::Fp128<C>* partials [[buffer(0)]],
    device jolt::Fp128<C>* out [[buffer(1)]],
    constant OneHotLimbParams& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    using F = jolt::Fp128<C>;
    ulong len = ulong(p.rings) * 512;
    if (i >= len) return;
    F sum = p.first_chunk == 0 ? F::zero() : out[i];
    for (uint part = 0; part < p.parts; part++) sum = sum + partials[ulong(part) * len + i];
    out[i] = sum;
}
} // namespace akita
