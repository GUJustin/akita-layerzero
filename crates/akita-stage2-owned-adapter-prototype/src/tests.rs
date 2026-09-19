use super::*;
pub(super) mod mock {
    use super::*;
    use std::cell::Cell;
    thread_local! {pub static LIVE:Cell<usize>=const {Cell::new(0)};pub static CALLS:Cell<usize>=const {Cell::new(0)};pub static FAIL:Cell<u32>=const {Cell::new(0)};}
    pub unsafe fn akita_stage2_admit(_: *const RawConfig, _: *const i8, _: u64) -> u32 {
        CALLS.set(CALLS.get() + 1);
        0
    }
    pub unsafe fn akita_stage2_create(
        _: *const RawConfig,
        _: *const i8,
        _: u64,
        out: *mut *mut c_void,
    ) -> u32 {
        CALLS.set(CALLS.get() + 1);
        unsafe { *out = std::ptr::null_mut() };
        if FAIL.get() == 7 {
            return 2;
        }
        unsafe { *out = Box::into_raw(Box::new(0u8)).cast() };
        LIVE.set(LIVE.get() + 1);
        0
    }
    pub unsafe fn akita_stage2_compact_entry(
        _: *mut c_void,
        _: *const RawRound,
        out: *mut Message,
    ) -> u32 {
        CALLS.set(CALLS.get() + 1);
        if FAIL.get() == 1 {
            return 3;
        }
        unsafe { *out = Message::default() };
        if FAIL.get() == 4 {
            unsafe {
                (*out).ordinary[0].c0 = PRIME;
            }
        }
        0
    }
    pub unsafe fn akita_stage2_advance(
        _: *mut c_void,
        _: *const RawRound,
        out: *mut Message,
    ) -> u32 {
        CALLS.set(CALLS.get() + 1);
        if FAIL.get() == 2 {
            return 3;
        }
        unsafe { *out = Message::default() };
        0
    }
    pub unsafe fn akita_stage2_finish(_: *mut c_void, out: *mut Element, n: u64) -> u32 {
        CALLS.set(CALLS.get() + 1);
        if FAIL.get() == 3 {
            return 3;
        }
        for i in 0..n as usize {
            unsafe { *out.add(i) = Element::default() };
        }
        0
    }
    pub unsafe fn akita_stage2_destroy(p: *mut c_void) -> u32 {
        if !p.is_null() {
            unsafe { drop(Box::from_raw(p.cast::<u8>())) };
            LIVE.set(LIVE.get() - 1);
        }
        0
    }
}
struct Fixture {
    alpha: Vec<Element>,
    weights: Vec<Element>,
    first: Vec<Element>,
    second: Vec<Element>,
    offsets: Vec<u64>,
}
impl Fixture {
    fn new(l: usize, c: usize, entry: bool) -> Self {
        let m = c / if entry { 4 } else { 2 };
        Self {
            alpha: vec![Element::default(); m],
            weights: vec![Element::default(); l],
            first: vec![Element::default(); 1],
            second: vec![Element::default(); l * m / 2],
            offsets: vec![0; l + 1],
        }
    }
    fn round(&self, c: usize) -> Round<'_> {
        Round {
            input_coefficients: c,
            skip_linear: false,
            r0: Element::default(),
            r1: Element::default(),
            alpha: &self.alpha,
            lane_weights: &self.weights,
            eq_first: &self.first,
            eq_second: &self.second,
            sources: &[],
            source_records: &[],
            lane_offsets: &self.offsets,
            references: &[],
            additional_live_len: self.weights.len() * self.alpha.len(),
            additional_domain_len: self.weights.len().next_power_of_two() * self.alpha.len(),
            binary_batching: Element::default(),
            additional_pairs: &[],
        }
    }
}
fn reset() {
    assert_eq!(mock::LIVE.get(), 0);
    mock::CALLS.set(0);
    mock::FAIL.set(0);
}
#[test]
fn admission_rejects_before_ffi_and_checks_padded_support() {
    reset();
    assert!(Element::from_limbs(PRIME, 0).is_err());
    assert!(Config::new(0, 16, 4, 1 << 20, 64).is_err());
    assert!(Config::new(3, 4, 4, 1 << 20, 16).is_err());
    let config = Config::new(3, 16, 4, 1 << 20, 64).unwrap();
    assert!(Owner::create(config, &[7; 48]).is_err());
    assert_eq!(mock::CALLS.get(), 0);
    let f = Fixture::new(3, 16, true);
    let mut r = f.round(16);
    let zero = Element::default();
    let padded = [AdditionalPair {
        parent: 7,
        linear0: zero,
        linear1: zero,
        binary0: zero,
        binary1: zero,
    }];
    r.additional_pairs = &padded;
    assert!(r.checked(3, 16, 64, true).is_ok());
    let outside = [AdditionalPair {
        parent: 8,
        ..padded[0]
    }];
    r.additional_pairs = &outside;
    assert!(r.checked(3, 16, 64, true).is_err());
    let duplicate = [padded[0], padded[0]];
    r.additional_pairs = &duplicate;
    assert!(r.checked(3, 16, 64, true).is_err());
    let mut owner = Owner::create(config, &[0; 48]).unwrap();
    let calls = mock::CALLS.get();
    r.additional_pairs = &outside;
    assert!(owner.compact_entry(&r).is_err());
    assert_eq!(mock::CALLS.get(), calls);
    assert_eq!(mock::LIVE.get(), 1);
    drop(owner);
    assert_eq!(mock::LIVE.get(), 0);
}
#[test]
fn entry_advance_finish_and_early_drop_release_once() {
    reset();
    let config = Config::new(3, 32, 8, 1 << 20, 128).unwrap();
    let mut owner = Owner::create(config, &[0; 96]).unwrap();
    let e = Fixture::new(3, 32, true);
    owner.compact_entry(&e.round(32)).unwrap();
    let calls = mock::CALLS.get();
    assert!(owner.compact_entry(&e.round(32)).is_err());
    assert_eq!(calls, mock::CALLS.get());
    let a = Fixture::new(3, 8, false);
    owner.advance(&a.round(8)).unwrap();
    assert_eq!(owner.finish().unwrap().len(), 12);
    assert_eq!(mock::LIVE.get(), 0);
    drop(Owner::create(config, &[0; 96]).unwrap());
    assert_eq!(mock::LIVE.get(), 0);
}
#[test]
fn all_native_failure_stages_close_and_noncanonical_output_rejected() {
    for failure in [1, 2, 3, 4, 7] {
        reset();
        let config = Config::new(3, 32, 4, 1 << 20, 128).unwrap();
        if failure == 7 {
            mock::FAIL.set(7);
            assert!(Owner::create(config, &[0; 96]).is_err());
            assert_eq!(mock::LIVE.get(), 0);
            continue;
        }
        let mut owner = Owner::create(config, &[0; 96]).unwrap();
        let e = Fixture::new(3, 32, true);
        if failure == 1 || failure == 4 {
            mock::FAIL.set(failure);
            assert!(owner.compact_entry(&e.round(32)).is_err());
            assert_eq!(mock::LIVE.get(), 0);
            assert_eq!(
                owner.compact_entry(&e.round(32)).unwrap_err(),
                Error::Closed
            );
        } else {
            owner.compact_entry(&e.round(32)).unwrap();
            mock::FAIL.set(failure);
            if failure == 2 {
                let a = Fixture::new(3, 8, false);
                assert!(owner.advance(&a.round(8)).is_err());
            } else {
                assert!(owner.finish().is_err());
                assert_eq!(mock::LIVE.get(), 0);
                continue;
            }
        }
        drop(owner);
        assert_eq!(mock::LIVE.get(), 0);
    }
    mock::FAIL.set(0);
}

#[test]
fn successor_ring_padding_survives_entry_and_advance() {
    reset();
    assert!(Config::new(1, 32, 4, 1 << 20, 16).is_err());
    assert!(Config::new(1, 32, 4, 1 << 20, 127).is_err());
    let mut owner = Owner::create(Config::new(1, 32, 4, 1 << 20, 128).unwrap(), &[0; 32]).unwrap();
    let e = Fixture::new(1, 32, true);
    let mut r = e.round(32);
    r.additional_domain_len = 32;
    let z = Element::default();
    let support = [AdditionalPair {
        parent: 15,
        linear0: z,
        linear1: z,
        binary0: z,
        binary1: z,
    }];
    r.additional_pairs = &support;
    owner.compact_entry(&r).unwrap();
    let a = Fixture::new(1, 8, false);
    let mut r = a.round(8);
    r.additional_domain_len = 16;
    owner.advance(&r).unwrap();
    assert_eq!(owner.finish().unwrap().len(), 4);
    assert_eq!(mock::LIVE.get(), 0);
}
#[test]
fn mock_error_paths_leave_output_sentinels_untouched() {
    reset();
    let sentinel = Element::from_limbs(17, 19).unwrap();
    let mut out = Message {
        ordinary: [sentinel; 6],
        additional: [sentinel; 4],
    };
    mock::FAIL.set(1);
    assert_eq!(
        unsafe {
            mock::akita_stage2_compact_entry(std::ptr::null_mut(), std::ptr::null(), &mut out)
        },
        3
    );
    assert_eq!(out.ordinary, [sentinel; 6]);
    assert_eq!(out.additional, [sentinel; 4]);
    mock::FAIL.set(2);
    assert_eq!(
        unsafe { mock::akita_stage2_advance(std::ptr::null_mut(), std::ptr::null(), &mut out) },
        3
    );
    assert_eq!(out.ordinary, [sentinel; 6]);
    assert_eq!(out.additional, [sentinel; 4]);
    mock::FAIL.set(3);
    let mut fields = [sentinel; 4];
    assert_eq!(
        unsafe { mock::akita_stage2_finish(std::ptr::null_mut(), fields.as_mut_ptr(), 4) },
        3
    );
    assert_eq!(fields, [sentinel; 4]);
    mock::FAIL.set(0);
}
