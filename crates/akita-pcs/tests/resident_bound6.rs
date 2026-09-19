//! Full PCS correctness against an unchanged application-owned trusted catalog.
//! The bundled immutable fixture preserves the application policy and mixed relation schedule.
#![cfg(feature = "resident-stage2-observer")]
use akita_config::{CommitmentConfig, RingDimensionScheduleMode};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, SelectedProverOpeningData, UniformProverStack,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    AkitaBatchedProof, BasisMode, DecompositionParams, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, SisModulusProfileId,
};
use jolt_field::{Ext2, ExtField, Prime64Offset59, Ring, Zero};
type F = Prime64Offset59;
type E = Ext2<F>;

// Exact application policy from post147 proof-core::configs::D128FullBound6.
// No catalog row or relation-mode override is introduced for this test.
#[derive(Clone)]
struct D128FullBound6;
impl CommitmentConfig for D128FullBound6 {
    type Field = F;
    type ExtField = E;
    const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
        RingDimensionScheduleMode::UniformDimension {
            ring_dimension: 128,
        };
    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 6,
            log_open_bound: Some(64),
        }
    }
    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        akita_config::proof_optimized::fp64::Dense::ring_challenge_config(d)
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q64Offset59
    }
    fn opening_basis_range() -> (u32, u32) {
        (3, 6)
    }
    fn inner_basis_range() -> (u32, u32) {
        (3, 11)
    }
    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        akita_types::sis::CommittedSourceClass::BalancedSignedDigit
    }
    fn schedule_family_name() -> &'static str {
        "aerie_fp64_d128_bound6"
    }
}
fn run() {
    let bytes = std::fs::read(
        std::env::var_os("AKITA_RESIDENT_TEST_CATALOG")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/resident_bound6.aks")
            }),
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<D128FullBound6>::from_schedule_artifact(&bytes)
        .expect("trusted original catalog");
    let nv = 14;
    let evals = (0..1usize << nv)
        .map(|i| F::from_i64((i % 8) as i64 - 4))
        .collect::<Vec<_>>();
    let poly = DensePoly::from_field_evals(nv, &evals).unwrap();
    let point = (0..nv)
        .map(|i| E::new(F::from_u64(3 + i as u64), F::from_u64(17 + 2 * i as u64)))
        .collect::<Vec<_>>();
    let weights = akita_types::lagrange_weights(&point).unwrap();
    let expected_opening = weights
        .iter()
        .zip(&evals)
        .fold(E::zero(), |a, (w, v)| a + *w * E::lift_base(*v));
    let setup = scheme.setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack =
        UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
            .unwrap();
    let verifier_setup = scheme.setup_verifier(&setup).unwrap();
    let committed = scheme
        .commit(
            &setup,
            std::slice::from_ref(&poly),
            stack.commitment(),
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let poly_refs = [&poly];
    let opening = || {
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::zero()],
            committed.committed_group.clone(),
        )
        .unwrap()])
        .unwrap();
        SelectedProverOpeningData::from_committed_claims::<D128FullBound6>(
            claims,
            vec![committed.prover_state.clone()],
            vec![&poly_refs[..]],
            scheme.schedules(),
        )
        .unwrap()
    };
    let selection = opening().selection();
    let label = b"post147-resident-bound6";
    assert!(akita_prover::stage2_observer::take().is_empty());
    let mut ct = AkitaTranscript::<F>::new(label);
    let cpu = scheme
        .batched_prove(&setup, opening(), &stack, &mut ct, BasisMode::Lagrange)
        .unwrap();
    assert!(akita_prover::stage2_observer::take().is_empty());
    let mut nt = AkitaTranscript::<F>::new(label);
    let native = scheme
        .batched_prove_hybrid_stage2(&setup, opening(), &stack, &mut nt, BasisMode::Lagrange)
        .unwrap();
    let events = akita_prover::stage2_observer::take();
    assert_eq!(events.len(), 3);
    assert_eq!(
        events.iter().map(|e| e.basis).collect::<Vec<_>>(),
        [8, 64, 64]
    );
    for (index, event) in events.iter().enumerate() {
        assert_eq!(event.level, index);
        if index < 2 {
            assert_eq!(event.declined, None);
            assert_eq!(event.compact_entries, 1);
            assert_eq!(event.exports, 1);
            assert!(event.columns >= 8);
            assert!(event.advances > 0);
        } else {
            assert_eq!(event.declined, Some("reduced-dense-preselected-cpu"));
            assert_eq!(
                (event.compact_entries, event.advances, event.exports),
                (0, 0, 0)
            );
        }
    }
    let mut cb = Vec::new();
    let mut nb = Vec::new();
    cpu.serialize_uncompressed(&mut cb).unwrap();
    native.serialize_uncompressed(&mut nb).unwrap();
    assert_eq!(cb, nb);
    let cpu_next = ct.challenge_scalar(b"after-proof");
    assert_eq!(cpu_next, nt.challenge_scalar(b"after-proof"));
    for proof_bytes in [&cb, &nb] {
        let proof =
            AkitaBatchedProof::<F, E>::deserialize_uncompressed(&proof_bytes[..], &cpu.shape())
                .unwrap();
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![expected_opening],
            &committed.committed_group,
        )
        .unwrap()])
        .unwrap();
        let mut vt = AkitaTranscript::<F>::new(label);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut vt,
                GroupBatchStatement::new(selection, claims).unwrap(),
                BasisMode::Lagrange,
            )
            .unwrap();
        assert_eq!(cpu_next, vt.challenge_scalar(b"after-proof"));
    }
    // The distinct strict API remains fail-closed; it must not silently become hybrid.
    let mut strict_t = AkitaTranscript::<F>::new(label);
    let result = scheme.batched_prove_resident_stage2(
        &setup,
        opening(),
        &stack,
        &mut strict_t,
        BasisMode::Lagrange,
    );
    assert!(
        matches!(result,Err(AkitaError::InvalidInput(ref text)) if text.contains("requires quotient-factored relation weights"))
    );
    let strict_events = akita_prover::stage2_observer::take();
    assert_eq!(strict_events.len(), 2);
    assert!(strict_events
        .iter()
        .all(|e| e.declined.is_none() && e.compact_entries == 1 && e.exports == 1));
}
#[test]
fn bound6_full_pcs_hybrid_matches_cpu_and_strict_rejects_reduced() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
