//! Constructor-derived nonzero CSR/Packing/additional coverage. No full-PCS claim.
use super::resident_owned::ResidentRelationProver;
use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_serialization::AkitaSerialize;
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::BasisMode;
use jolt_field::{Ext2, One, Prime64Offset59};
type B = Prime64Offset59;
type F = Ext2<B>;
fn value(i: usize) -> F {
    F::new(
        B::from_u64(31 + i as u64 * 13),
        B::from_u64(17 + i as u64 * 7),
    )
}
fn sample(t: &mut AkitaTranscript<B>) -> Result<F, AkitaError> {
    Ok(F::new(
        t.challenge_scalar(b"rho"),
        t.challenge_scalar(b"rho"),
    ))
}
fn sparse(lanes: usize, c: usize) -> PreparedProverLinearTerms<F> {
    // Public checked test constructor lowers nonzero source planes to Sparse CSR.
    let values = (0..lanes * c)
        .map(|i| if i % 5 == 0 { F::zero() } else { value(i) })
        .collect();
    let prepared = PreparedProverLinearTerms::from_dense(values, lanes, c);
    assert!(matches!(
        &prepared.lane_weights,
        evaluation_trace::PreparedLaneWeights::Sparse(_)
    ));
    prepared
}
fn fixture(
    basis: usize,
    lanes: usize,
    c: usize,
    linear: PreparedProverLinearTerms<F>,
    additional_mode: usize,
    dense: bool,
    binary_digits: bool,
) -> RelationRangeImageProver<F> {
    let lane_bits = lanes.next_power_of_two().trailing_zeros() as usize;
    let bits = c.trailing_zeros() as usize;
    let domain = (1 << lane_bits) * c;
    let digits = (0..lanes * c)
        .map(|i| {
            if binary_digits {
                -(i as i8 & 1)
            } else {
                ((i * 7) % basis) as i8 - (basis / 2) as i8
            }
        })
        .collect::<Vec<_>>();
    let point = (0..lane_bits + bits).map(value).collect::<Vec<_>>();
    let eq = EqPolynomial::evals(&point).unwrap();
    let alpha = (0..c).map(|i| value(i + 19)).collect::<Vec<_>>();
    let weights = (0..1 << lane_bits)
        .map(|i| value(i + 59))
        .collect::<Vec<_>>();
    let mut range = F::zero();
    let mut relation = F::zero();
    let mut linear_claim = F::zero();
    for (i, &d) in digits.iter().enumerate() {
        let w = F::from_i64(i64::from(d));
        range += eq[i] * w * (w + F::one());
        relation += w * alpha[i % c] * weights[i / c];
        linear_claim += w * linear.get(i / c, i % c, c);
    }
    assert!(linear.materialize_dense().iter().any(|v| !v.is_zero()));
    let compact = PackedSignedDigits::from_i8_digits_auto(digits);
    let additional = if additional_mode == 0 {
        None
    } else {
        let mut terms = Vec::new();
        if additional_mode & 1 != 0 {
            // Duplicate support is merged by the production constructor; the final
            // coordinate exercises padded zero witness reads when lanes are ragged.
            terms = vec![
                (0, value(101)),
                (0, value(102)),
                (c - 1, value(103)),
                (domain - 1, value(104)),
            ];
        }
        let intervals = if additional_mode & 2 != 0 {
            vec![0..c + 1, domain - 2..domain]
        } else {
            vec![]
        };
        let extra = AdditionalRelationTerms::new(
            &compact,
            domain,
            terms.clone(),
            &intervals,
            &point,
            value(111),
        )
        .unwrap();
        let independent = terms.iter().fold(F::zero(), |acc, (i, t)| {
            acc + *t * F::from_i64(i64::from(compact.get(*i).unwrap_or(0)))
        }) + intervals.iter().flat_map(|interval| interval.clone()).fold(
            F::zero(),
            |acc, i| {
                let w = F::from_i64(i64::from(compact.get(i).unwrap_or(0)));
                acc + value(111) * eq[i] * w * (w + F::one())
            },
        );
        assert_eq!(extra.input_claim(), independent);
        Some(extra)
    };
    let relation_weights = if dense {
        RelationWeightOracle::ReducedDense(
            DenseRelationWeights::new(
                (0..domain).map(|i| alpha[i % c] * weights[i / c]).collect(),
                lanes * c,
            )
            .unwrap(),
        )
    } else {
        RelationWeightOracle::QuotientFactored(
            RelationWeightFactorization::new(alpha, weights).unwrap(),
        )
    };
    RelationRangeImageProver::new(
        value(71),
        compact,
        &point,
        range,
        basis,
        relation_weights,
        lanes,
        lane_bits,
        bits,
        relation,
        linear,
        linear_claim,
        additional,
    )
    .unwrap()
}
fn parity(cpu: RelationRangeImageProver<F>, native: RelationRangeImageProver<F>) {
    let mut cpu = cpu;
    let mut native = ResidentRelationProver::new(native).unwrap();
    let mut ct = AkitaTranscript::<B>::prover(b"resident-branches", b"fixture");
    let mut nt = AkitaTranscript::<B>::prover(b"resident-branches", b"fixture");
    let expected =
        akita_sumcheck::prove_sumcheck::<B, _, F, _, _>(&mut cpu, &mut ct, sample).unwrap();
    let actual =
        akita_sumcheck::prove_fallible_sumcheck::<B, _, F, _, _>(&mut native, &mut nt, sample)
            .unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    expected.0.serialize_compressed(&mut a).unwrap();
    actual.0.serialize_compressed(&mut b).unwrap();
    assert_eq!(a, b);
    assert_eq!(expected.1, actual.1);
    assert_eq!(expected.2, actual.2);
    let completed = native.into_completed().unwrap();
    assert_eq!(cpu.final_w_eval(), completed.final_w_eval());
    assert_eq!(
        cpu.expected_final_claim().unwrap(),
        completed.expected_final_claim().unwrap()
    );
    assert_eq!(sample(&mut ct).unwrap(), sample(&mut nt).unwrap());
}
#[test]
fn resident_sparse_additional_constructor_states_match_cpu() {
    for basis in [4, 8, 16, 32, 64] {
        for mode in 0..4 {
            for binary in [false, true] {
                let make = || fixture(basis, 3, 16, sparse(3, 16), mode, false, binary);
                parity(make(), make());
            }
        }
    }
}
#[test]
fn resident_authenticated_packing_constructor_states_match_cpu() {
    for basis_mode in [BasisMode::Lagrange, BasisMode::Monomial] {
        for basis in [4, 8, 64] {
            for mode in [0, 3] {
                let make = || {
                    let (linear, lanes, c) =
                        coefficient_packing_terms::tests::resident_packing_fixture(basis_mode);
                    assert!(matches!(
                        &linear.lane_weights,
                        evaluation_trace::PreparedLaneWeights::Packing(_)
                    ));
                    fixture(basis, lanes, c, linear, mode, false, false)
                };
                parity(make(), make());
            }
        }
    }
}
#[test]
fn resident_reduced_dense_rejects_before_transcript_absorption() {
    let make = || fixture(8, 3, 16, sparse(3, 16), 3, true, false);
    // Establish this is a valid CPU state, not malformed input masquerading as
    // the unsupported representation admission test.
    let mut cpu = make();
    let mut cpu_t = AkitaTranscript::<B>::prover(b"resident-reject", b"fixture");
    akita_sumcheck::prove_sumcheck::<B, _, F, _, _>(&mut cpu, &mut cpu_t, sample).unwrap();
    let mut actual = AkitaTranscript::<B>::prover(b"resident-reject", b"fixture");
    let mut expected = AkitaTranscript::<B>::prover(b"resident-reject", b"fixture");
    let mut calls = 0;
    let result = (|| {
        let mut resident = ResidentRelationProver::new(make())?;
        akita_sumcheck::prove_fallible_sumcheck::<B, _, F, _, _>(&mut resident, &mut actual, |t| {
            calls += 1;
            sample(t)
        })
    })();
    assert!(
        matches!(result,Err(AkitaError::InvalidInput(ref text)) if text.contains("requires quotient-factored relation weights"))
    );
    assert_eq!(calls, 0);
    assert_eq!(sample(&mut actual).unwrap(), sample(&mut expected).unwrap());
}
