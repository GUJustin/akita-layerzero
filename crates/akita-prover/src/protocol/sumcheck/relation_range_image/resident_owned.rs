//! Default-off local owner integration. No transcript access and no CPU replay.
use super::evaluation_trace::resident_stage2_descriptors::Limits;
use super::*;
use akita_stage2_owned_adapter_prototype::{
    AdditionalPair, Config, Element, Message, Owner, Reference, Round, Source,
};
use akita_sumcheck::FallibleSumcheckInstanceProver;
use jolt_field::{Ext2, Prime64Offset59, Ring, Zero};
type F = Ext2<Prime64Offset59>;
const CAP: usize = 1 << 26;
const DESCRIPTORS: usize = 1 << 20;
fn invalid(message: &str) -> AkitaError {
    AkitaError::InvalidInput(message.into())
}
fn native(error: akita_stage2_owned_adapter_prototype::Error) -> AkitaError {
    invalid(&format!("resident Stage2: {error:?}"))
}
fn encode(x: F) -> Result<Element, AkitaError> {
    Element::from_limbs(x.c0().to_canonical_u64(), x.c1().to_canonical_u64()).map_err(native)
}
fn limbs(x: [u64; 2]) -> Result<Element, AkitaError> {
    Element::from_limbs(x[0], x[1]).map_err(native)
}
fn decode(x: Element) -> F {
    let [a, b] = x.limbs();
    F::new(Prime64Offset59::from_u64(a), Prime64Offset59::from_u64(b))
}
fn encode_slice(x: &[F]) -> Result<Vec<Element>, AkitaError> {
    x.iter().copied().map(encode).collect()
}
const LIMITS: Limits = Limits {
    source_count: DESCRIPTORS,
    source_elements: CAP,
    references: DESCRIPTORS,
    lanes: CAP,
};
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Prefix,
    Resident,
    Tail,
    Failed,
    Finished,
}

/// Owns the concrete thread-local native session and the original CPU scalar state.
/// Construction performs static admission and compact upload before the driver
/// absorbs the Stage2 claim. Native execution starts only at the second challenge.
/// On failure, the wrapper is terminal and releases its owner; callers must abort
/// the proof attempt. CPU tail handoff is an exact export, never a replay.
pub(crate) struct ResidentRelationProver {
    cpu: RelationRangeImageProver<F>,
    owner: Option<Owner>,
    phase: Phase,
    first: Option<F>,
    initial_coefficients: usize,
    deferred_prefix: bool,
    additional: Option<UniPoly<F>>,
    awaiting_challenge: bool,
    #[cfg(feature = "resident-stage2-observer")]
    counts: [usize; 3],
}
impl ResidentRelationProver {
    pub(crate) fn new(cpu: RelationRangeImageProver<F>) -> Result<Self, AkitaError> {
        let c = cpu
            .common_alpha_factor()
            .ok_or_else(|| invalid("resident requires quotient-factored relation weights"))?
            .len();
        let l = cpu.live_lane_count;
        let domain = 1usize
            .checked_shl(u32::try_from(cpu.num_vars).map_err(|_| invalid("domain bits"))?)
            .ok_or_else(|| invalid("domain overflow"))?;
        let live = l.checked_mul(c).ok_or_else(|| invalid("live overflow"))?;
        if cpu.rounds_completed != 0
            || !c.is_power_of_two()
            || c < 8
            || ![4, 8, 16, 32, 64].contains(&cpu.b)
            || live > domain
            || domain > CAP
        {
            return Err(invalid("resident Stage2 initial geometry/phase"));
        }
        let compact = match &cpu.witness_state {
            WitnessState::CompactPrefix(v) if v.len() == live => v,
            _ => return Err(invalid("resident compact witness")),
        };
        // Pre-transcript representation admission. This currently copies the
        // factored source planes once; it does not expand a lane×coefficient t.
        let descriptor = cpu.linear_terms.serialize_resident(LIMITS)?;
        if descriptor.lanes != l as u64 || descriptor.coefficients != c as u64 {
            return Err(invalid("initial CSR geometry"));
        }
        if let Some(additional) = &cpu.additional_relation_terms {
            let wire = additional.serialize_resident_additional(live, DESCRIPTORS)?;
            if wire.domain_len != domain as u64 {
                return Err(invalid("initial additional domain"));
            }
        }
        drop(descriptor);
        let config = Config::new(l, c, cpu.b, 8 << 30, domain).map_err(native)?;
        // Upstream stores compact digits packed; decode canonically, never reinterpret.
        // This temporary signed-byte plane and native copy are charged to admission.
        let decoded: Vec<i8> = compact.iter().collect();
        let owner = Owner::create(config, &decoded).map_err(native)?;
        let deferred_prefix = cpu.using_deferred_compact_prefix();
        Ok(Self {
            cpu,
            owner: Some(owner),
            phase: Phase::Prefix,
            first: None,
            initial_coefficients: c,
            deferred_prefix,
            additional: None,
            awaiting_challenge: false,
            #[cfg(feature = "resident-stage2-observer")]
            counts: [0; 3],
        })
    }
    /// Return the fully folded CPU state only after successful driver finalization.
    /// This preserves the outer fold's existing expected_final_claim/final_w_eval
    /// checks without exposing or duplicating their oracle equations.
    pub(crate) fn into_completed(self) -> Result<RelationRangeImageProver<F>, AkitaError> {
        if self.phase != Phase::Finished
            || self.owner.is_some()
            || self.awaiting_challenge
            || self.cpu.rounds_completed != self.cpu.num_vars
        {
            return Err(invalid("resident result before successful finalization"));
        }
        match &self.cpu.witness_state {
            WitnessState::FoldedSuffix(v) if v.len() == 1 => Ok(self.cpu),
            _ => Err(invalid("resident final witness shape")),
        }
    }
    #[cfg(feature = "resident-stage2-observer")]
    pub(crate) fn execution_counts(&self) -> [usize; 3] {
        self.counts
    }
    fn alpha(&self) -> Result<&[F], AkitaError> {
        self.cpu
            .common_alpha_factor()
            .ok_or_else(|| invalid("resident quotient alpha"))
    }
    fn fold_alpha(&mut self, r: F) -> Result<(), AkitaError> {
        match &mut self.cpu.relation_state {
            RelationRoundState::QuotientFactored { weights, .. } => {
                fold_evals_in_place(weights.components_mut().0, r);
                Ok(())
            }
            _ => Err(invalid("resident quotient alpha")),
        }
    }
    fn poison(&mut self) {
        self.owner = None;
        self.additional = None;
        self.phase = Phase::Failed;
    }
    fn bind_host(&mut self, r: F) {
        if let Some(additional) = &mut self.cpu.additional_relation_terms {
            additional.bind(r);
        }
        if let Some(norm) = self.cpu.prev_norm_poly.take() {
            self.cpu.prev_norm_claim = norm.evaluate(&r);
        }
        self.cpu.split_eq.bind(r);
    }
    fn dispatch(&mut self, input_c: usize, r0: F, r1: F, entry: bool) -> Result<(), AkitaError> {
        let l = self.cpu.live_lane_count;
        let c = self.alpha()?.len();
        let live = l
            .checked_mul(c)
            .ok_or_else(|| invalid("next live overflow"))?;
        let bits = self
            .cpu
            .num_vars
            .checked_sub(self.cpu.rounds_completed)
            .ok_or_else(|| invalid("round domain"))?;
        let domain = 1usize
            .checked_shl(u32::try_from(bits).map_err(|_| invalid("round domain bits"))?)
            .ok_or_else(|| invalid("round domain overflow"))?;
        let wire = self.cpu.linear_terms.serialize_resident(LIMITS)?;
        if wire.lanes != l as u64 || wire.coefficients != c as u64 {
            return Err(invalid("round CSR geometry"));
        }
        let sources = wire
            .sources
            .into_iter()
            .map(limbs)
            .collect::<Result<Vec<_>, _>>()?;
        let records = wire
            .source_records
            .into_iter()
            .map(|[offset, lanes]| Source { offset, lanes })
            .collect::<Vec<_>>();
        let references = wire
            .references
            .into_iter()
            .map(|[source, lane, a, b]| {
                Ok(Reference {
                    source,
                    lane,
                    factor: limbs([a, b])?,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let (beta, pairs) = if let Some(additional) = &self.cpu.additional_relation_terms {
            let a = additional.serialize_resident_additional(live, DESCRIPTORS)?;
            if a.domain_len != domain as u64 || a.live_len != live as u64 {
                return Err(invalid("round additional geometry"));
            }
            (
                limbs(a.binary_batching)?,
                a.pairs
                    .into_iter()
                    .map(|p| {
                        Ok(AdditionalPair {
                            parent: p[0],
                            linear0: limbs([p[1], p[2]])?,
                            linear1: limbs([p[3], p[4]])?,
                            binary0: limbs([p[5], p[6]])?,
                            binary1: limbs([p[7], p[8]])?,
                        })
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()?,
            )
        } else {
            (Element::default(), Vec::new())
        };
        let alpha = encode_slice(self.alpha()?)?;
        let weights = encode_slice(
            self.cpu
                .relation_lane_weights()
                .ok_or_else(|| invalid("quotient lane weights"))?
                .get(..l)
                .ok_or_else(|| invalid("lane weights"))?,
        )?;
        let (first, second) = self.cpu.split_eq.remaining_eq_tables();
        let pair_count = live / 2;
        if first.is_empty() || !first.len().is_power_of_two() || first.len() > pair_count {
            return Err(invalid("equality first extent"));
        }
        let second_len = pair_count.div_ceil(first.len());
        let first = encode_slice(first)?;
        let second = encode_slice(
            second
                .get(..second_len)
                .ok_or_else(|| invalid("equality second extent"))?,
        )?;
        let skip = self.cpu.can_skip_norm_linear_coeff();
        let request = Round {
            input_coefficients: input_c,
            skip_linear: skip,
            r0: encode(r0)?,
            r1: encode(r1)?,
            alpha: &alpha,
            lane_weights: &weights,
            eq_first: &first,
            eq_second: &second,
            sources: &sources,
            source_records: &records,
            lane_offsets: &wire.lane_offsets,
            references: &references,
            additional_domain_len: domain,
            additional_live_len: live,
            binary_batching: beta,
            additional_pairs: &pairs,
        };
        let owner = self
            .owner
            .as_mut()
            .ok_or_else(|| invalid("missing resident owner"))?;
        let Message {
            ordinary,
            additional,
        } = if entry {
            owner.compact_entry(&request)
        } else {
            owner.advance(&request)
        }
        .map_err(native)?;
        #[cfg(feature = "resident-stage2-observer")]
        {
            self.counts[usize::from(!entry)] += 1;
        }
        let norm = if skip {
            NormRoundTerms::SkipLinear([decode(ordinary[0]), decode(ordinary[2])])
        } else {
            NormRoundTerms::Full([
                decode(ordinary[0]),
                decode(ordinary[1]),
                decode(ordinary[2]),
            ])
        };
        // Keep the ordinary norm polynomial separate from the additional cubic.
        self.cpu.cached_round_poly = Some(self.cpu.combine_terms(
            norm,
            [
                decode(ordinary[3]),
                decode(ordinary[4]),
                decode(ordinary[5]),
            ],
        ));
        let mut extra = additional.map(decode).to_vec();
        akita_algebra::poly::trim_trailing_zeros(&mut extra);
        self.additional = Some(UniPoly::from_coeffs(extra));
        if c == 2 {
            let values = self
                .owner
                .take()
                .ok_or_else(|| invalid("missing export owner"))?
                .finish()
                .map_err(native)?;
            if values.len() != live {
                return Err(invalid("resident export length"));
            }
            #[cfg(feature = "resident-stage2-observer")]
            {
                self.counts[2] += 1;
            }
            self.cpu.witness_state =
                WitnessState::FoldedSuffix(values.into_iter().map(decode).collect());
            // The ordinary cached message is retained; the existing CPU path
            // computes the additional term from the exported initialized witness.
            self.additional = None;
            self.phase = Phase::Tail;
        } else {
            self.phase = Phase::Resident;
        }
        Ok(())
    }
    fn ingest_inner(&mut self, round: usize, r: F) -> Result<(), AkitaError> {
        if !self.awaiting_challenge || round != self.cpu.rounds_completed {
            return Err(invalid("resident ingest order"));
        }
        match self.phase {
            Phase::Tail => SumcheckInstanceProver::ingest_challenge(&mut self.cpu, round, r),
            Phase::Prefix if round == 0 => {
                SumcheckInstanceProver::ingest_challenge(&mut self.cpu, round, r);
                self.first = Some(r);
            }
            Phase::Prefix if round == 1 => {
                let r0 = self
                    .first
                    .take()
                    .ok_or_else(|| invalid("missing first challenge"))?;
                self.bind_host(r);
                if self.deferred_prefix {
                    let alpha =
                        RelationRangeImageProver::fold_alpha_two_rounds(self.alpha()?, r0, r);
                    self.cpu.linear_terms.fold_two_coefficients(r0, r);
                    self.cpu.replace_common_alpha_factor(alpha);
                } else {
                    // Wider bases retained the original CPU first fold and its
                    // initialized N/2 witness. Fold only the second challenge's
                    // host factors here; native entry still consumes original
                    // compact input with both challenges. This CPU work and
                    // temporary allocation remain part of the proof lifecycle.
                    self.cpu.fold_linear_terms_for_current_round(r);
                    self.fold_alpha(r)?;
                }
                self.cpu.rounds_completed += 1;
                if let RelationRoundState::QuotientFactored { prefix, .. } =
                    &mut self.cpu.relation_state
                {
                    *prefix = QuotientPrefixState::Disabled;
                } else {
                    return Err(invalid("resident relation representation changed"));
                }
                // The compact or CPU first-fold allocation is released after the
                // owner already copied it at admission. Empty state is private
                // to this wrapper and never read by the CPU while resident.
                self.cpu.witness_state = WitnessState::FoldedSuffix(Vec::new());
                self.dispatch(self.initial_coefficients, r0, r, true)?;
            }
            Phase::Resident => {
                let c = self.alpha()?.len();
                if c < 4 || !self.cpu.in_coefficient_round() {
                    return Err(invalid("resident coefficient phase"));
                }
                self.bind_host(r);
                self.cpu.fold_linear_terms_for_current_round(r);
                self.fold_alpha(r)?;
                self.cpu.rounds_completed += 1;
                self.dispatch(c, r, F::zero(), false)?;
            }
            _ => return Err(invalid("resident terminal phase")),
        }
        self.awaiting_challenge = false;
        Ok(())
    }
}
impl FallibleSumcheckInstanceProver<F> for ResidentRelationProver {
    fn num_rounds(&self) -> usize {
        self.cpu.num_vars
    }
    fn degree_bound(&self) -> usize {
        3
    }
    fn input_claim(&self) -> F {
        self.cpu.input_claim
    }
    fn compute_round_univariate(
        &mut self,
        round: usize,
        claim: F,
    ) -> Result<UniPoly<F>, AkitaError> {
        let result = (|| {
            if self.awaiting_challenge
                || round != self.cpu.rounds_completed
                || round >= self.cpu.num_vars
            {
                return Err(invalid("resident compute order"));
            }
            let poly = match self.phase {
                Phase::Prefix | Phase::Tail => {
                    SumcheckInstanceProver::compute_round_univariate(&mut self.cpu, round, claim)
                }
                Phase::Resident => {
                    let mut p = self
                        .cpu
                        .cached_round_poly
                        .take()
                        .ok_or_else(|| invalid("missing ordinary message"))?;
                    let extra = self
                        .additional
                        .take()
                        .ok_or_else(|| invalid("missing additional message"))?;
                    p.coeffs
                        .resize(p.coeffs.len().max(extra.coeffs.len()), F::zero());
                    for (i, x) in extra.coeffs.into_iter().enumerate() {
                        p.coeffs[i] += x;
                    }
                    p
                }
                _ => return Err(invalid("resident terminal compute")),
            };
            self.awaiting_challenge = true;
            Ok(poly)
        })();
        if result.is_err() {
            self.poison();
        }
        result
    }
    fn ingest_challenge(&mut self, round: usize, r: F) -> Result<(), AkitaError> {
        let result = self.ingest_inner(round, r);
        if result.is_err() {
            self.poison();
        }
        result
    }
    fn finalize(&mut self) -> Result<(), AkitaError> {
        if self.phase != Phase::Tail
            || self.owner.is_some()
            || self.awaiting_challenge
            || self.cpu.rounds_completed != self.cpu.num_vars
        {
            self.poison();
            return Err(invalid("resident finalize state"));
        }
        SumcheckInstanceProver::finalize(&mut self.cpu);
        self.phase = Phase::Finished;
        Ok(())
    }
}
