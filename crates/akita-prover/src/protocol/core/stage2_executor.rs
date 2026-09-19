//! Typed local Stage2 selection. Native errors are fatal; there is no CPU replay.
use super::*;
#[derive(Clone, Copy)]
pub(super) struct Stage2Context {
    pub level: usize,
    pub basis: usize,
    pub columns: usize,
    pub lanes: usize,
    pub domain: usize,
    pub compression_layers: usize,
    pub negative_binary_intervals: usize,
}
type Completed<E> = ((SumcheckProof<E>, Vec<E>, E), RelationRangeImageProver<E>);
pub(super) trait Stage2Executor<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
{
    fn prove<T, S>(
        &self,
        prover: RelationRangeImageProver<E>,
        transcript: &mut T,
        context: Stage2Context,
        sample: S,
    ) -> Result<Completed<E>, AkitaError>
    where
        T: Transcript<F>,
        S: FnMut(&mut T) -> Result<E, AkitaError>;
}
pub(super) struct CpuStage2;
impl<F, E> Stage2Executor<F, E> for CpuStage2
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + Unreduced + Fold + Ring + AkitaSerialize,
{
    fn prove<T, S>(
        &self,
        mut prover: RelationRangeImageProver<E>,
        transcript: &mut T,
        context: Stage2Context,
        sample: S,
    ) -> Result<Completed<E>, AkitaError>
    where
        T: Transcript<F>,
        S: FnMut(&mut T) -> Result<E, AkitaError>,
    {
        let _ = (
            context.level,
            context.basis,
            context.columns,
            context.lanes,
            context.domain,
            context.compression_layers,
            context.negative_binary_intervals,
        );
        let output = prove_sumcheck::<F, T, E, _, _>(&mut prover, transcript, sample)?;
        Ok((output, prover))
    }
}
#[cfg(feature = "resident-stage2-owned")]
pub(super) struct ResidentStage2;
#[cfg(feature = "resident-stage2-owned")]
impl Stage2Executor<jolt_field::Prime64Offset59, jolt_field::Ext2<jolt_field::Prime64Offset59>>
    for ResidentStage2
{
    fn prove<T, S>(
        &self,
        prover: RelationRangeImageProver<jolt_field::Ext2<jolt_field::Prime64Offset59>>,
        transcript: &mut T,
        context: Stage2Context,
        sample: S,
    ) -> Result<Completed<jolt_field::Ext2<jolt_field::Prime64Offset59>>, AkitaError>
    where
        T: Transcript<jolt_field::Prime64Offset59>,
        S: FnMut(&mut T) -> Result<jolt_field::Ext2<jolt_field::Prime64Offset59>, AkitaError>,
    {
        use crate::protocol::sumcheck::relation_range_image::resident_owned::ResidentRelationProver;
        if ![4, 8, 16, 32, 64].contains(&context.basis)
            || context.columns < 8
            || !context.columns.is_power_of_two()
            || context.domain > 1 << 26
        {
            return Err(AkitaError::InvalidInput(
                "resident Stage2 unsupported geometry; no CPU fallback".into(),
            ));
        }
        // Static admission and owner upload precede the Stage2 claim absorption.
        let mut owner = ResidentRelationProver::new(prover)?;
        let output = akita_sumcheck::prove_fallible_sumcheck::<
            jolt_field::Prime64Offset59,
            T,
            jolt_field::Ext2<jolt_field::Prime64Offset59>,
            _,
            _,
        >(&mut owner, transcript, sample)?;
        #[cfg(feature = "resident-stage2-observer")]
        let counts = owner.execution_counts();
        let completed = owner.into_completed()?;
        #[cfg(feature = "resident-stage2-observer")]
        crate::stage2_observer::record(crate::stage2_observer::Event {
            level: context.level,
            basis: context.basis,
            columns: context.columns,
            lanes: context.lanes,
            domain: context.domain,
            compression_layers: context.compression_layers,
            negative_binary_intervals: context.negative_binary_intervals,
            declined: None,
            compact_entries: counts[0],
            advances: counts[1],
            exports: counts[2],
        })?;
        Ok((output, completed))
    }
}

/// Explicit static composition, not recovery from native admission or execution errors.
#[cfg(feature = "resident-stage2-owned")]
pub(super) struct HybridStage2;
#[cfg(feature = "resident-stage2-owned")]
impl Stage2Executor<jolt_field::Prime64Offset59, jolt_field::Ext2<jolt_field::Prime64Offset59>>
    for HybridStage2
{
    fn prove<T, S>(
        &self,
        prover: RelationRangeImageProver<jolt_field::Ext2<jolt_field::Prime64Offset59>>,
        transcript: &mut T,
        context: Stage2Context,
        sample: S,
    ) -> Result<Completed<jolt_field::Ext2<jolt_field::Prime64Offset59>>, AkitaError>
    where
        T: Transcript<jolt_field::Prime64Offset59>,
        S: FnMut(&mut T) -> Result<jolt_field::Ext2<jolt_field::Prime64Offset59>, AkitaError>,
    {
        // Decide from the actual relation representation before either driver
        // absorbs its claim. All errors from the chosen route propagate unchanged.
        if prover.is_quotient_factored() {
            return ResidentStage2.prove(prover, transcript, context, sample);
        }
        let completed = CpuStage2.prove(prover, transcript, context, sample)?;
        #[cfg(feature = "resident-stage2-observer")]
        crate::stage2_observer::record(crate::stage2_observer::Event {
            level: context.level,
            basis: context.basis,
            columns: context.columns,
            lanes: context.lanes,
            domain: context.domain,
            compression_layers: context.compression_layers,
            negative_binary_intervals: context.negative_binary_intervals,
            declined: Some("reduced-dense-preselected-cpu"),
            compact_entries: 0,
            advances: 0,
            exports: 0,
        })?;
        Ok(completed)
    }
}
