//! Checked canonical fp64/Ext2 transport. No full lane-by-coefficient table.
//! Lane CSR preserves contribution order. Sources are concatenated once, and
//! each reference carries a source index, source lane and factor. This is an owned
//! canonical serialization copy, not a zero-copy native upload.
use super::*;
use akita_error::checked;
use jolt_field::{Ext2, Prime64Offset59, Ring, Zero};
type F = Ext2<Prime64Offset59>;

/// Caller limits bound allocation before any output buffer is created.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) source_elements: usize,
    pub(crate) source_count: usize,
    pub(crate) references: usize,
    pub(crate) lanes: usize,
}

/// All integers are canonical u64; native consumers must validate again.
/// `references` entries are [source index, source lane, factor.c0, factor.c1].
/// `source_records` entries are [element offset, lane count].
/// `lane_offsets` has lanes+1 entries and indexes references, not bytes.
/// Sources and offsets use the CURRENT coefficient count after host binding.
pub(crate) struct Serialized {
    pub(crate) lanes: u64,
    pub(crate) coefficients: u64,
    pub(crate) sources: Vec<[u64; 2]>,
    pub(crate) lane_offsets: Vec<u64>,
    pub(crate) source_records: Vec<[u64; 2]>,
    pub(crate) references: Vec<[u64; 4]>,
}
fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid resident Stage2 linear descriptors".into())
}
fn canonical(value: F) -> Result<[u64; 2], AkitaError> {
    Ok([value.c0().to_canonical_u64(), value.c1().to_canonical_u64()])
}
fn reserve<T>(n: usize) -> Result<Vec<T>, AkitaError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| invalid())?;
    Ok(v)
}

impl PreparedProverLinearTerms<F> {
    /// Serialize coefficient-phase support only. No transcript/state mutation;
    /// malformed maps and dense lane-phase state fail before output allocation.
    pub(crate) fn serialize_resident(&self, limits: Limits) -> Result<Serialized, AkitaError> {
        let c = self.coeff_count;
        let lanes = self.live_lane_count;
        if c < 2 || !c.is_power_of_two() || lanes == 0 || lanes > limits.lanes {
            return Err(invalid());
        }
        self.validate_len(checked::product([lanes, c]).ok_or_else(invalid)?)?;
        let total =
            checked::sum(self.sources.iter().map(|s| s.values.len())).ok_or_else(invalid)?;
        if self.sources.len() > limits.source_count || total > limits.source_elements {
            return Err(invalid());
        }
        // Verify every contribution, including Packing's lookup map, so the
        // transported sequence is exactly the CPU for_each_lane_source sequence.
        let check = |source: usize, lane: usize| -> Result<(), AkitaError> {
            let s = self.sources.get(source).ok_or_else(invalid)?;
            if lane >= s.lane_count {
                return Err(invalid());
            }
            Ok(())
        };
        let refs = match &self.lane_weights {
            PreparedLaneWeights::Sparse(terms) => {
                for lane in terms {
                    for t in lane {
                        check(t.source_index, t.lane)?;
                        canonical(t.factor)?;
                    }
                }
                checked::sum(terms.iter().map(Vec::len)).ok_or_else(invalid)?
            }
            PreparedLaneWeights::Packing(map) => {
                for segment in &map.segments {
                    if segment.lane_count == 0 {
                        return Err(invalid());
                    }
                    let end = segment
                        .target_lane_start
                        .checked_add(segment.lane_count)
                        .ok_or_else(invalid)?;
                    let source_end = segment
                        .source_lane_start
                        .checked_add(segment.lane_count)
                        .ok_or_else(invalid)?;
                    if end > lanes {
                        return Err(invalid());
                    }
                    check(segment.source_index, source_end - 1)?;
                    canonical(segment.factor)?;
                }
                let expected_count =
                    checked::sum(map.segments.iter().map(|s| s.lane_count)).ok_or_else(invalid)?;
                if expected_count > limits.references || map.lane_to_segment.len() != lanes {
                    return Err(invalid());
                }
                let mut actual_count = 0usize;
                for (lane, encoded) in map.lane_to_segment.iter().enumerate() {
                    if let Some(encoded) = encoded {
                        let segment = map.segments.get(encoded.get() - 1).ok_or_else(invalid)?;
                        if lane < segment.target_lane_start
                            || lane - segment.target_lane_start >= segment.lane_count
                        {
                            return Err(invalid());
                        }
                        actual_count = actual_count.checked_add(1).ok_or_else(invalid)?;
                    }
                }
                if actual_count != expected_count {
                    return Err(invalid());
                }
                expected_count
            }
            PreparedLaneWeights::Dense(_) => return Err(invalid()),
        };
        if refs > limits.references {
            return Err(invalid());
        }
        let offsets_len = lanes.checked_add(1).ok_or_else(invalid)?;
        checked::product([total, 16]).ok_or_else(invalid)?;
        checked::product([self.sources.len(), 16]).ok_or_else(invalid)?;
        checked::product([refs, 32]).ok_or_else(invalid)?;
        checked::product([offsets_len, 8]).ok_or_else(invalid)?;
        for source in &self.sources {
            for &x in &source.values {
                canonical(x)?;
            }
        }
        // Output allocation starts only after complete shape, budget and field admission.
        let mut sources = reserve(total)?;
        let mut source_records = reserve(self.sources.len())?;
        for source in &self.sources {
            source_records.push([
                u64::try_from(sources.len()).map_err(|_| invalid())?,
                u64::try_from(source.lane_count).map_err(|_| invalid())?,
            ]);
            for &x in &source.values {
                sources.push(canonical(x)?);
            }
        }
        let mut lane_offsets = reserve(offsets_len)?;
        let mut references = reserve(refs)?;
        for lane in 0..lanes {
            lane_offsets.push(u64::try_from(references.len()).map_err(|_| invalid())?);
            let mut add =
                |factor: F, source: usize, source_lane: usize| -> Result<(), AkitaError> {
                    let factor = canonical(factor)?;
                    references.push([
                        u64::try_from(source).map_err(|_| invalid())?,
                        u64::try_from(source_lane).map_err(|_| invalid())?,
                        factor[0],
                        factor[1],
                    ]);
                    Ok(())
                };
            match &self.lane_weights {
                PreparedLaneWeights::Sparse(terms) => {
                    for t in &terms[lane] {
                        add(t.factor, t.source_index, t.lane)?;
                    }
                }
                PreparedLaneWeights::Packing(map) => {
                    if let Some(encoded) = map.lane_to_segment[lane] {
                        let s = &map.segments[encoded.get() - 1];
                        add(
                            s.factor,
                            s.source_index,
                            s.source_lane_start + lane - s.target_lane_start,
                        )?;
                    }
                }
                PreparedLaneWeights::Dense(_) => return Err(invalid()),
            }
        }
        lane_offsets.push(u64::try_from(references.len()).map_err(|_| invalid())?);
        Ok(Serialized {
            lanes: u64::try_from(lanes).map_err(|_| invalid())?,
            coefficients: u64::try_from(c).map_err(|_| invalid())?,
            sources,
            source_records,
            lane_offsets,
            references,
        })
    }
}
