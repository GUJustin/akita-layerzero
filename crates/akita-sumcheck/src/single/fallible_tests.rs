use super::*;
use crate::FallibleSumcheckInstanceProver;
use akita_algebra::uni_poly::UniPoly;
use akita_transcript::AkitaTranscript;
use jolt_field::{Ext2, One, Prime64Offset59, Ring, Zero};
use std::{cell::Cell, rc::Rc};

type Base = Prime64Offset59;
type E = Ext2<Base>;
type Tr = AkitaTranscript<Base>;

fn tr() -> Tr {
    Tr::prover(labels::DOMAIN_AKITA_PROTOCOL, b"resident-port")
}
fn sample(t: &mut Tr) -> Result<E, AkitaError> {
    Ok(E::new(
        t.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        t.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
    ))
}
fn poly(round: usize, claim: E) -> UniPoly<E> {
    let c = E::new(
        Base::from_u64(3 + round as u64),
        Base::from_u64(9 + round as u64),
    );
    UniPoly::from_coeffs(vec![E::zero(), claim - c, c])
}
#[derive(Clone)]
struct Cpu {
    rounds: usize,
    bound: usize,
    events: Vec<String>,
}
impl Cpu {
    fn new(rounds: usize) -> Self {
        Self {
            rounds,
            bound: 2,
            events: vec![],
        }
    }
}
impl SumcheckInstanceProver<E> for Cpu {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        self.bound
    }
    fn input_claim(&self) -> E {
        E::new(Base::from_u64(17), Base::from_u64(29))
    }
    fn compute_round_univariate(&mut self, round: usize, claim: E) -> UniPoly<E> {
        self.events.push(format!("compute:{round}"));
        poly(round, claim)
    }
    fn ingest_challenge(&mut self, round: usize, _: E) {
        self.events.push(format!("ingest:{round}"));
    }
    fn finalize(&mut self) {
        self.events.push("finalize".into());
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fail {
    Never,
    Compute(usize),
    Ingest(usize),
    Finish,
}
struct Local {
    cpu: Cpu,
    failure: Fail,
    // This real field prevents both Send and Sync. The generic driver must not
    // accidentally recover the old trait's thread-safety constraints.
    calls: Rc<Cell<usize>>,
}
impl FallibleSumcheckInstanceProver<E> for Local {
    fn num_rounds(&self) -> usize {
        self.cpu.rounds
    }
    fn degree_bound(&self) -> usize {
        self.cpu.bound
    }
    fn input_claim(&self) -> E {
        self.cpu.input_claim()
    }
    fn compute_round_univariate(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<UniPoly<E>, AkitaError> {
        self.calls.set(self.calls.get() + 1);
        let g = self.cpu.compute_round_univariate(round, claim);
        if self.failure == Fail::Compute(round) {
            return Err(AkitaError::InvalidInput("injected compute".into()));
        }
        Ok(g)
    }
    fn ingest_challenge(&mut self, round: usize, rho: E) -> Result<(), AkitaError> {
        self.calls.set(self.calls.get() + 1);
        self.cpu.ingest_challenge(round, rho);
        if self.failure == Fail::Ingest(round) {
            return Err(AkitaError::InvalidInput("injected ingest".into()));
        }
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), AkitaError> {
        self.calls.set(self.calls.get() + 1);
        self.cpu.finalize();
        if self.failure == Fail::Finish {
            return Err(AkitaError::InvalidInput("injected finish".into()));
        }
        Ok(())
    }
}
fn bytes(proof: &SumcheckProof<E>) -> Vec<u8> {
    let mut out = vec![];
    proof.serialize_compressed(&mut out).unwrap();
    out
}

#[test]
fn fallible_local_and_infallible_adapter_match_original_transcript() {
    for rounds in [0, 1, 4] {
        let mut old = Cpu::new(rounds);
        let mut cpu = old.clone();
        let calls = Rc::new(Cell::new(0));
        let mut local = Local {
            cpu: old.clone(),
            failure: Fail::Never,
            calls: calls.clone(),
        };
        let (mut ot, mut ct, mut lt) = (tr(), tr(), tr());
        let expected = standard_original_test_oracle::original_prove::<Base, _, E, _, _>(
            &mut old, &mut ot, sample,
        )
        .unwrap();
        let actual = prove_sumcheck::<Base, _, E, _, _>(&mut cpu, &mut ct, sample).unwrap();
        let native =
            prove_fallible_sumcheck::<Base, _, E, _, _>(&mut local, &mut lt, sample).unwrap();
        assert_eq!(bytes(&expected.0), bytes(&actual.0));
        assert_eq!(bytes(&expected.0), bytes(&native.0));
        assert_eq!(expected, actual);
        assert_eq!(expected, native);
        let next = sample(&mut ot);
        assert_eq!(next, sample(&mut ct));
        assert_eq!(next, sample(&mut lt));
        assert_eq!(old.events, cpu.events);
        assert_eq!(old.events, local.cpu.events);
        assert_eq!(calls.get(), rounds * 2 + 1);
    }
}

/// Independent transcript prefix oracle: append the claim, and then exactly
/// the successful compute/degree-admitted messages, without calling the driver.
fn expected_stop(failure: Fail, bound: usize) -> (Tr, Vec<String>, usize) {
    let mut t = tr();
    let mut claim = Cpu::new(3).input_claim();
    t.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    let mut events = vec![];
    let mut calls = 0;
    for round in 0..3 {
        events.push(format!("compute:{round}"));
        calls += 1;
        if failure == Fail::Compute(round) {
            return (t, events, calls);
        }
        let g = poly(round, claim).compress();
        if g.degree() > bound {
            return (t, events, calls);
        }
        t.append_serde(labels::ABSORB_SUMCHECK_ROUND, &g);
        let rho = sample(&mut t).unwrap();
        claim = g.eval_from_hint(&claim, &rho);
        events.push(format!("ingest:{round}"));
        calls += 1;
        if failure == Fail::Ingest(round) {
            return (t, events, calls);
        }
    }
    events.push("finalize".into());
    calls += 1;
    (t, events, calls)
}
#[test]
fn fallible_errors_stop_at_exact_compute_ingest_and_finish_boundary() {
    for failure in [
        Fail::Compute(0),
        Fail::Compute(1),
        Fail::Compute(2),
        Fail::Ingest(0),
        Fail::Ingest(1),
        Fail::Ingest(2),
        Fail::Finish,
        Fail::Never,
    ] {
        let bound = if failure == Fail::Never { 1 } else { 2 };
        let mut local = Local {
            cpu: Cpu {
                rounds: 3,
                bound,
                events: vec![],
            },
            failure,
            calls: Rc::new(Cell::new(0)),
        };
        let mut actual = tr();
        let error = prove_fallible_sumcheck::<Base, _, E, _, _>(&mut local, &mut actual, sample)
            .unwrap_err();
        let (mut expected, events, calls) = expected_stop(failure, bound);
        assert_eq!(sample(&mut actual), sample(&mut expected));
        assert_eq!(local.cpu.events, events);
        assert_eq!(local.calls.get(), calls);
        let message = error.to_string();
        match failure {
            Fail::Compute(_) => assert!(message.contains("injected compute")),
            Fail::Ingest(_) => assert!(message.contains("injected ingest")),
            Fail::Finish => assert!(message.contains("injected finish")),
            Fail::Never => assert!(message.contains("exceeds bound")),
        }
    }
    // The legacy adapter must preserve the original degree-failure prefix too.
    let (mut old, mut new) = (
        Cpu {
            rounds: 3,
            bound: 1,
            events: vec![],
        },
        Cpu {
            rounds: 3,
            bound: 1,
            events: vec![],
        },
    );
    let (mut ot, mut nt) = (tr(), tr());
    assert!(
        standard_original_test_oracle::original_prove::<Base, _, E, _, _>(
            &mut old, &mut ot, sample
        )
        .is_err()
    );
    assert!(prove_sumcheck::<Base, _, E, _, _>(&mut new, &mut nt, sample).is_err());
    assert_eq!(old.events, new.events);
    assert_eq!(sample(&mut ot), sample(&mut nt));
}

#[test]
fn fallible_sampling_error_preserves_upstream_message_boundary() {
    let mut local = Local {
        cpu: Cpu::new(3),
        failure: Fail::Never,
        calls: Rc::new(Cell::new(0)),
    };
    let mut actual = tr();
    let mut samples = 0usize;
    assert!(
        prove_fallible_sumcheck::<Base, _, E, _, _>(&mut local, &mut actual, |t| {
            samples += 1;
            if samples == 2 {
                t.append_bytes(b"failed-sample", b"attempt");
                return Err(AkitaError::InvalidInput("sampling failure".into()));
            }
            sample(t)
        })
        .is_err()
    );
    let mut expected = tr();
    let mut claim = Cpu::new(3).input_claim();
    expected.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    let p0 = poly(0, claim).compress();
    expected.append_serde(labels::ABSORB_SUMCHECK_ROUND, &p0);
    let r0 = sample(&mut expected).unwrap();
    claim = p0.eval_from_hint(&claim, &r0);
    expected.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly(1, claim).compress());
    expected.append_bytes(b"failed-sample", b"attempt");
    assert_eq!(sample(&mut actual), sample(&mut expected));
    assert_eq!(local.cpu.events, vec!["compute:0", "ingest:0", "compute:1"]);
    assert_eq!(samples, 2);
}
