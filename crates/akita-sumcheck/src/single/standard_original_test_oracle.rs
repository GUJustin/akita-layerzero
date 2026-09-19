use super::*;
/// Prove one standard sumcheck instance.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub(super) fn original_prove<F, T, E, S, P>(
    prover: &mut P,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
    E: Field + AkitaSerialize,
    S: FnMut(&mut T) -> Result<E, AkitaError>,
    P: SumcheckInstanceProver<E> + ?Sized,
{
    let num_rounds = prover.num_rounds();
    let mut claim = prover.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = prover.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);
    for round in 0..num_rounds {
        let _round_span = tracing::info_span!(
            "sumcheck_round",
            round,
            table_len = 1usize << (num_rounds - round)
        )
        .entered();
        let poly = {
            let _span = tracing::info_span!("sumcheck_round_univariate").entered();
            prover.compute_round_univariate(round, claim)
        };
        debug_assert_eq!(
            poly.evaluate(&E::zero()) + poly.evaluate(&E::one()),
            claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );
        let compressed = poly.compress();
        if compressed.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let challenge = sample_challenge(transcript)?;
        claim = compressed.eval_from_hint(&claim, &challenge);
        {
            let _span = tracing::info_span!("sumcheck_round_fold").entered();
            prover.ingest_challenge(round, challenge);
        }
        challenges.push(challenge);
        round_polys.push(compressed);
    }
    prover.finalize();
    Ok((SumcheckProof { round_polys }, challenges, claim))
}
