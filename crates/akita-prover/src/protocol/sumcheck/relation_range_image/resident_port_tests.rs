//! Actual upstream packed-digit state versus the pinned upstream CPU oracle.
use super::resident_owned::ResidentRelationProver;
use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_serialization::AkitaSerialize;
use akita_transcript::{AkitaTranscript, Transcript};
use jolt_field::{Ext2, One, Prime64Offset59};
type B = Prime64Offset59;
type F = Ext2<B>;
fn value(i: usize) -> F {
    F::new(
        B::from_u64(u64::MAX - 1024 - i as u64),
        B::from_u64(17 + 31 * i as u64),
    )
}
fn fixture(basis: usize, coefficient_bits: usize) -> RelationRangeImageProver<F> {
    let lanes = 3usize;
    let lane_bits = 2usize;
    let c = 1usize << coefficient_bits;
    let digits = (0..lanes * c)
        .map(|i| ((i * 7) % basis) as i8 - (basis / 2) as i8)
        .collect::<Vec<_>>();
    let point = (0..lane_bits + coefficient_bits)
        .map(value)
        .collect::<Vec<_>>();
    let eq = EqPolynomial::evals(&point).unwrap();
    let alpha = (0..c).map(|i| value(i + 19)).collect::<Vec<_>>();
    let weights = (0..1 << lane_bits)
        .map(|i| value(i + 59))
        .collect::<Vec<_>>();
    let mut range = F::zero();
    let mut relation = F::zero();
    for (i, &d) in digits.iter().enumerate() {
        let w = F::from_i64(i64::from(d));
        range += eq[i] * w * (w + F::one());
        relation += w * alpha[i % c] * weights[i / c];
    }
    RelationRangeImageProver::new(
        value(71),
        PackedSignedDigits::from_i8_digits_auto(digits),
        &point,
        range,
        basis,
        RelationWeightOracle::QuotientFactored(
            RelationWeightFactorization::new(alpha, weights).unwrap(),
        ),
        lanes,
        lane_bits,
        coefficient_bits,
        relation,
        PreparedProverLinearTerms::zero(lanes, c),
        F::zero(),
        None,
    )
    .unwrap()
}
fn sample(t: &mut AkitaTranscript<B>) -> Result<F, AkitaError> {
    Ok(F::new(
        t.challenge_scalar(b"stage2-rho"),
        t.challenge_scalar(b"stage2-rho"),
    ))
}
#[test]
fn resident_upstream_packed_digits_match_cpu_full_stage2() {
    for basis in [4, 8, 16, 32, 64] {
        for bits in [3, 4] {
            let mut cpu = fixture(basis, bits);
            let mut resident = ResidentRelationProver::new(fixture(basis, bits)).unwrap();
            let mut ct = AkitaTranscript::<B>::prover(b"resident-upstream", b"fixture");
            let mut nt = AkitaTranscript::<B>::prover(b"resident-upstream", b"fixture");
            let expected =
                akita_sumcheck::prove_sumcheck::<B, _, F, _, _>(&mut cpu, &mut ct, sample).unwrap();
            let actual = akita_sumcheck::prove_fallible_sumcheck::<B, _, F, _, _>(
                &mut resident,
                &mut nt,
                sample,
            )
            .unwrap();
            let mut eb = Vec::new();
            let mut ab = Vec::new();
            expected.0.serialize_compressed(&mut eb).unwrap();
            actual.0.serialize_compressed(&mut ab).unwrap();
            assert_eq!(eb, ab);
            assert_eq!(expected.1, actual.1);
            assert_eq!(expected.2, actual.2);
            let completed = resident.into_completed().unwrap();
            assert_eq!(cpu.final_w_eval(), completed.final_w_eval());
            assert_eq!(
                cpu.expected_final_claim().unwrap(),
                completed.expected_final_claim().unwrap()
            );
            assert_eq!(sample(&mut ct).unwrap(), sample(&mut nt).unwrap());
        }
    }
}
