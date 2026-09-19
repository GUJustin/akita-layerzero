#ifndef AKITA_STAGE2_OWNED_H
#define AKITA_STAGE2_OWNED_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Synchronous, same-thread ABI. No exception crosses this boundary.
 * Every nonempty input span is readable for its count; spans are borrowed only
 * until return. Native code copies inputs before submission. Never aliases the
 * resident witness with caller storage. All output spans must be disjoint from
 * inputs/owner storage. The Rust wrapper guarantees these pointer contracts.
 * Native still validates every count, canonical limb, CSR/digit/domain bound.
 */
typedef struct { uint64_t c0,c1; } AkitaStage2Ext;
typedef struct { uint64_t offset,lanes; } AkitaStage2Source;
typedef struct { uint64_t source,lane; AkitaStage2Ext factor; } AkitaStage2Reference;
typedef struct {
    uint64_t parent;
    AkitaStage2Ext linear0,linear1,binary0,binary1;
} AkitaStage2AdditionalPair;
typedef struct {
    uint64_t lanes,coefficients,basis,max_payload_bytes,initial_domain_len;
} AkitaStage2Config;
typedef struct {
    uint64_t input_coefficients,skip_linear;
    AkitaStage2Ext r0,r1;
    const AkitaStage2Ext *alpha; uint64_t alpha_len;
    const AkitaStage2Ext *lane_weights; uint64_t lane_weights_len;
    const AkitaStage2Ext *eq_first; uint64_t eq_first_len;
    const AkitaStage2Ext *eq_second; uint64_t eq_second_len;
    const AkitaStage2Ext *sources; uint64_t sources_len;
    const AkitaStage2Source *source_records; uint64_t source_records_len;
    const uint64_t *lane_offsets; uint64_t lane_offsets_len;
    const AkitaStage2Reference *references; uint64_t references_len;
    uint64_t additional_domain_len,additional_live_len;
    AkitaStage2Ext binary_batching;
    const AkitaStage2AdditionalPair *additional_pairs; uint64_t additional_pairs_len;
} AkitaStage2Round;
typedef struct {
    AkitaStage2Ext ordinary[6];
    AkitaStage2Ext additional[4];
} AkitaStage2Message;
typedef struct AkitaStage2Owner AkitaStage2Owner;
enum {
    AKITA_STAGE2_OK=0,
    AKITA_STAGE2_INVALID=1,
    AKITA_STAGE2_ALLOCATION=2,
    AKITA_STAGE2_DEVICE=3,
    AKITA_STAGE2_STATE=4,
    AKITA_STAGE2_THREAD=5,
    AKITA_STAGE2_INTERNAL=6
};
/* Admission is pure: no Metal work, allocations or global mutation. Requires
 * C>=8 power of two, L>0, L*C<=2^26, basis4/8 and each balanced signed digit.
 * initial_domain_len is power of two, >=L*C, <=2^26; successor-ring padding
 * may exceed next_pow2(L*C) and is preserved explicitly through each fold.
 * Native conservative payload bounds must fit max_payload_bytes<=8GiB.
 * Rust invokes admission/create BEFORE Stage2 claim absorption.
 */
uint32_t akita_stage2_admit(const AkitaStage2Config*,const int8_t*,uint64_t);
/* *out is always initialized to null; failure releases all partial resources.
 * Success owns copied compact input, pipeline/queue/buffers, creating-thread ID.
 */
uint32_t akita_stage2_create(const AkitaStage2Config*,const int8_t*,uint64_t,AkitaStage2Owner** out);
/* entry: ReadyCompact -> ReadyCoefficients, C -> C/4. r0 then r1 bind low bits.
 * advance: ReadyCoefficients -> ReadyCoefficients, C -> C/2; r1 MUST be zero.
 * Input alpha/linear/equality describe OUTPUT witness/next ordinary message.
 * additional_live_len = L*outputC, additional_domain_len = stored domain / (entry?4:2).
 * Sorted unique additional parents may read zero-padded coordinates up to that
 * domain; each missing side is represented by zero fields. Empty pair spans OK.
 * Only six ordinary fields feed host norm reconstruction (skip middle is zero).
 * Additional cubic stays separate and never enters host prev_norm_poly.
 * Complete admission precedes writes/submission and leaves owner intact on INVALID.
 * Any allocation/encoder/device/error after operation starts poisons the owner.
 * Output message remains untouched on failure. Publish only after wait+validation.
 */
uint32_t akita_stage2_compact_entry(AkitaStage2Owner*,const AkitaStage2Round*,AkitaStage2Message*);
uint32_t akita_stage2_advance(AkitaStage2Owner*,const AkitaStage2Round*,AkitaStage2Message*);
/* Exact initialized current L*C fields only. Successful finish consumes state;
 * count mismatch rejects before output writes and leaves state intact.
 * Failure after readback starts poisons; caller discards the complete output.
 */
uint32_t akita_stage2_finish(AkitaStage2Owner*,AkitaStage2Ext* out,uint64_t count);
/* null is a no-op success. Wrong thread returns THREAD without destroying.
 * Correct-thread destruction releases resources in an autorelease pool; no throw.
 */
uint32_t akita_stage2_destroy(AkitaStage2Owner*);
#ifdef __cplusplus
}
#endif
#endif
