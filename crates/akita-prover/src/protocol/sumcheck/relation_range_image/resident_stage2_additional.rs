//! Canonical sparse cubic descriptors; zero-padded witness indices are legal.
use super::*;
use akita_error::checked;
use jolt_field::{Ext2, Prime64Offset59, Ring, Zero};
type F = Ext2<Prime64Offset59>;
pub(crate) struct AdditionalDescriptor {
    pub(crate) domain_len: u64,
    pub(crate) live_len: u64,
    pub(crate) binary_batching: [u64; 2],
    /// [parent, l0.c0,l0.c1,l1.c0,l1.c1,b0.c0,b0.c1,b1.c0,b1.c1].
    pub(crate) pairs: Vec<[u64; 9]>,
}
fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid resident additional terms".into())
}
fn limbs(x: F) -> Result<[u64; 2], AkitaError> {
    Ok([x.c0().to_canonical_u64(), x.c1().to_canonical_u64()])
}
impl AdditionalRelationTerms<F> {
    /// No mutation/writes before admission. max_pairs explicitly bounds allocation.
    /// Must be called AFTER this round's host bind, against current native witness.
    pub(crate) fn serialize_resident_additional(
        &self,
        live_len: usize,
        max_pairs: usize,
    ) -> Result<AdditionalDescriptor, AkitaError> {
        if self.domain_len < 2 || !self.domain_len.is_power_of_two() || live_len > self.domain_len {
            return Err(invalid());
        }
        let beta = limbs(self.binary_batching)?;
        let mut previous = None;
        let mut parent = None;
        let mut count = 0usize;
        for weight in &self.weights {
            if weight.index >= self.domain_len || previous.is_some_and(|i| i >= weight.index) {
                return Err(invalid());
            }
            limbs(weight.linear)?;
            limbs(weight.binary)?;
            let next = weight.index / 2;
            if parent != Some(next) {
                count = count.checked_add(1).ok_or_else(invalid)?;
                parent = Some(next);
            }
            previous = Some(weight.index);
        }
        if count > max_pairs {
            return Err(invalid());
        }
        checked::product([count, 72]).ok_or_else(invalid)?;
        let mut pairs: Vec<[u64; 9]> = Vec::new();
        pairs.try_reserve_exact(count).map_err(|_| invalid())?;
        for weight in &self.weights {
            let parent = u64::try_from(weight.index / 2).map_err(|_| invalid())?;
            if pairs.last().is_none_or(|p| p[0] != parent) {
                pairs.push([parent, 0, 0, 0, 0, 0, 0, 0, 0]);
            }
            let p = pairs.last_mut().ok_or_else(invalid)?;
            let side = weight.index % 2;
            let linear = limbs(weight.linear)?;
            let binary = limbs(weight.binary)?;
            p[1 + 2 * side..3 + 2 * side].copy_from_slice(&linear);
            p[5 + 2 * side..7 + 2 * side].copy_from_slice(&binary);
        }
        Ok(AdditionalDescriptor {
            domain_len: u64::try_from(self.domain_len).map_err(|_| invalid())?,
            live_len: u64::try_from(live_len).map_err(|_| invalid())?,
            binary_batching: beta,
            pairs,
        })
    }
}
