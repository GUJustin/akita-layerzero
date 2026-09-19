// Experimental exact arithmetic shared by a Metal feasibility probe and its
// host tests. This is not linked into the aggregate prover.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdint>
using ulong = uint64_t;
using uint = uint32_t;
#endif

struct Wide { ulong lo, hi; };
struct Ext { ulong c0, c1; };
struct LookupInput { Ext p0, p1, q0, q1, lambda, weight; };
struct ProbeOutput { Ext sum, difference, product, lookup; };
struct LookupParams { Ext lambda; uint pairs, width; };

// Four 32x32 products, with every intermediate bounded below 2^64.
inline Wide wide_mul(ulong a, ulong b) {
    ulong a0 = uint(a), a1 = a >> 32, b0 = uint(b), b1 = b >> 32;
    ulong p00 = a0*b0, p01 = a0*b1, p10 = a1*b0, p11 = a1*b1;
    ulong middle = (p00 >> 32) + ulong(uint(p01)) + ulong(uint(p10));
    return Wide{(middle << 32) | ulong(uint(p00)),
                p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32)};
}

inline ulong field_add(ulong a, ulong b) {
    const ulong prime = 0xffffffffffffffc5ul;
    ulong sum = a+b;
    // Canonical operands imply the carry case's low word is <= 2^64-120;
    // adding 59 cannot overflow again and is already below the prime.
    if (sum < a) return sum+59;
    return sum >= prime ? sum-prime : sum;
}

inline ulong field_sub(ulong a, ulong b) {
    const ulong prime = 0xffffffffffffffc5ul;
    return a >= b ? a-b : prime-(b-a);
}

inline ulong field_mul(ulong a, ulong b) {
    const ulong prime = 0xffffffffffffffc5ul;
    Wide product = wide_mul(a,b);
    Wide folded = wide_mul(product.hi,59);
    ulong low = folded.lo+product.lo;
    ulong high = folded.hi+ulong(low < folded.lo); // <= 59
    ulong settled = low+high*59;
    // A carry leaves a small low word; this correction cannot overflow.
    if (settled < low) settled += 59;
    return settled >= prime ? settled-prime : settled;
}

inline Ext ext_add(Ext a, Ext b) {
    return Ext{field_add(a.c0,b.c0),field_add(a.c1,b.c1)};
}
inline Ext ext_sub(Ext a, Ext b) {
    return Ext{field_sub(a.c0,b.c0),field_sub(a.c1,b.c1)};
}
inline Ext ext_mul(Ext a, Ext b) {
    ulong ac=field_mul(a.c0,b.c0), bd=field_mul(a.c1,b.c1);
    return Ext{field_add(ac,field_add(bd,bd)),
               field_add(field_mul(a.c0,b.c1),field_mul(a.c1,b.c0))};
}

inline Ext lookup_relation(Ext p0, Ext p1, Ext q0, Ext q1, Ext lambda) {
    return ext_add(ext_add(ext_mul(p0,q1),ext_mul(p1,q0)),
                   ext_mul(lambda,ext_mul(q0,q1)));
}

inline ProbeOutput probe(LookupInput x) {
    Ext h=lookup_relation(x.p0,x.p1,x.q0,x.q1,x.lambda);
    return ProbeOutput{ext_add(x.p0,x.q0), ext_sub(x.p0,x.q0),
                       ext_mul(x.p0,x.q0), ext_mul(x.weight,h)};
}
