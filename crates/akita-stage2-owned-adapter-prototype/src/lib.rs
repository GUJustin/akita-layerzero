//! Prototype checked Rust boundary for stage2.h. No challenger/proof serialization.
//! On Apple Silicon, build the bundled native library or use an explicit archive override.
use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};
const PRIME: u64 = u64::MAX - 58;
const MAX_N: usize = 1 << 26;
const MAX_DESCRIPTORS: usize = 1 << 20;
const MAX_BYTES: u64 = 8 << 30;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Admission(&'static str),
    Allocation,
    Native(u32),
    Closed,
    NoncanonicalOutput,
}
type Result<T> = std::result::Result<T, Error>;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Element {
    c0: u64,
    c1: u64,
}
impl Element {
    pub fn from_limbs(c0: u64, c1: u64) -> Result<Self> {
        if c0 >= PRIME || c1 >= PRIME {
            return Err(Error::Admission("noncanonical field"));
        }
        Ok(Self { c0, c1 })
    }
    pub fn limbs(self) -> [u64; 2] {
        [self.c0, self.c1]
    }
    fn canonical(self) -> bool {
        self.c0 < PRIME && self.c1 < PRIME
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Source {
    pub offset: u64,
    pub lanes: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub source: u64,
    pub lane: u64,
    pub factor: Element,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AdditionalPair {
    pub parent: u64,
    pub linear0: Element,
    pub linear1: Element,
    pub binary0: Element,
    pub binary1: Element,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct RawConfig {
    lanes: u64,
    coefficients: u64,
    basis: u64,
    max_payload_bytes: u64,
    initial_domain_len: u64,
}
#[derive(Clone, Copy)]
pub struct Config {
    raw: RawConfig,
}
impl Config {
    pub fn new(
        lanes: usize,
        coefficients: usize,
        basis: usize,
        max_payload_bytes: u64,
        initial_domain_len: usize,
    ) -> Result<Self> {
        if lanes == 0
            || !coefficients.is_power_of_two()
            || coefficients < 8
            || ![4, 8, 16, 32, 64].contains(&basis)
            || max_payload_bytes == 0
            || max_payload_bytes > MAX_BYTES
        {
            return Err(Error::Admission("configuration"));
        }
        let n = lanes
            .checked_mul(coefficients)
            .ok_or(Error::Admission("size overflow"))?;
        if n > MAX_N
            || !initial_domain_len.is_power_of_two()
            || initial_domain_len < n
            || initial_domain_len > MAX_N
        {
            return Err(Error::Admission("witness capacity"));
        }
        Ok(Self {
            raw: RawConfig {
                lanes: to_u64(lanes)?,
                coefficients: to_u64(coefficients)?,
                basis: to_u64(basis)?,
                max_payload_bytes,
                initial_domain_len: to_u64(initial_domain_len)?,
            },
        })
    }
    fn validate_compact(self, compact: &[i8]) -> Result<()> {
        let n = self
            .raw
            .lanes
            .checked_mul(self.raw.coefficients)
            .ok_or(Error::Admission("size overflow"))?;
        if to_u64(compact.len())? != n
            || compact.iter().any(|&x| {
                i64::from(x) < -(self.raw.basis as i64 / 2)
                    || i64::from(x) >= self.raw.basis as i64 / 2
            })
        {
            return Err(Error::Admission("compact digits/length"));
        }
        Ok(())
    }
}
/// All factors describe the next witness, after the caller has advanced its host
/// alpha/equality/linear/additional state exactly once. No input witness upload.
pub struct Round<'a> {
    pub input_coefficients: usize,
    pub skip_linear: bool,
    pub r0: Element,
    pub r1: Element,
    pub alpha: &'a [Element],
    pub lane_weights: &'a [Element],
    pub eq_first: &'a [Element],
    pub eq_second: &'a [Element],
    pub sources: &'a [Element],
    pub source_records: &'a [Source],
    pub lane_offsets: &'a [u64],
    pub references: &'a [Reference],
    pub additional_domain_len: usize,
    pub additional_live_len: usize,
    pub binary_batching: Element,
    pub additional_pairs: &'a [AdditionalPair],
}
#[repr(C)]
struct RawRound {
    input_coefficients: u64,
    skip_linear: u64,
    r0: Element,
    r1: Element,
    alpha: *const Element,
    alpha_len: u64,
    lane_weights: *const Element,
    lane_weights_len: u64,
    eq_first: *const Element,
    eq_first_len: u64,
    eq_second: *const Element,
    eq_second_len: u64,
    sources: *const Element,
    sources_len: u64,
    source_records: *const Source,
    source_records_len: u64,
    lane_offsets: *const u64,
    lane_offsets_len: u64,
    references: *const Reference,
    references_len: u64,
    additional_domain_len: u64,
    additional_live_len: u64,
    binary_batching: Element,
    additional_pairs: *const AdditionalPair,
    additional_pairs_len: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Message {
    pub ordinary: [Element; 6],
    pub additional: [Element; 4],
}
#[cfg(not(test))]
unsafe extern "C" {
    fn akita_stage2_admit(config: *const RawConfig, values: *const i8, count: u64) -> u32;
    fn akita_stage2_create(
        config: *const RawConfig,
        values: *const i8,
        count: u64,
        out: *mut *mut c_void,
    ) -> u32;
    fn akita_stage2_compact_entry(
        owner: *mut c_void,
        round: *const RawRound,
        out: *mut Message,
    ) -> u32;
    fn akita_stage2_advance(owner: *mut c_void, round: *const RawRound, out: *mut Message) -> u32;
    fn akita_stage2_finish(owner: *mut c_void, out: *mut Element, count: u64) -> u32;
    fn akita_stage2_destroy(owner: *mut c_void) -> u32;
}
const _: () = {
    assert!(std::mem::size_of::<Element>() == 16 && std::mem::align_of::<Element>() == 8);
    assert!(std::mem::offset_of!(Element, c0) == 0);
    assert!(std::mem::offset_of!(Element, c1) == 8);
    assert!(std::mem::size_of::<Source>() == 16 && std::mem::align_of::<Source>() == 8);
    assert!(std::mem::offset_of!(Source, offset) == 0);
    assert!(std::mem::offset_of!(Source, lanes) == 8);
    assert!(std::mem::size_of::<Reference>() == 32 && std::mem::align_of::<Reference>() == 8);
    assert!(std::mem::offset_of!(Reference, source) == 0);
    assert!(std::mem::offset_of!(Reference, lane) == 8);
    assert!(std::mem::offset_of!(Reference, factor) == 16);
    assert!(
        std::mem::size_of::<AdditionalPair>() == 72 && std::mem::align_of::<AdditionalPair>() == 8
    );
    assert!(std::mem::offset_of!(AdditionalPair, parent) == 0);
    assert!(std::mem::offset_of!(AdditionalPair, linear0) == 8);
    assert!(std::mem::offset_of!(AdditionalPair, linear1) == 24);
    assert!(std::mem::offset_of!(AdditionalPair, binary0) == 40);
    assert!(std::mem::offset_of!(AdditionalPair, binary1) == 56);
    assert!(std::mem::size_of::<RawConfig>() == 40 && std::mem::align_of::<RawConfig>() == 8);
    assert!(std::mem::offset_of!(RawConfig, lanes) == 0);
    assert!(std::mem::offset_of!(RawConfig, coefficients) == 8);
    assert!(std::mem::offset_of!(RawConfig, basis) == 16);
    assert!(std::mem::offset_of!(RawConfig, max_payload_bytes) == 24);
    assert!(std::mem::offset_of!(RawConfig, initial_domain_len) == 32);
    assert!(std::mem::size_of::<RawRound>() == 224 && std::mem::align_of::<RawRound>() == 8);
    assert!(std::mem::offset_of!(RawRound, input_coefficients) == 0);
    assert!(std::mem::offset_of!(RawRound, skip_linear) == 8);
    assert!(std::mem::offset_of!(RawRound, r0) == 16);
    assert!(std::mem::offset_of!(RawRound, r1) == 32);
    assert!(std::mem::offset_of!(RawRound, alpha) == 48);
    assert!(std::mem::offset_of!(RawRound, alpha_len) == 56);
    assert!(std::mem::offset_of!(RawRound, lane_weights) == 64);
    assert!(std::mem::offset_of!(RawRound, lane_weights_len) == 72);
    assert!(std::mem::offset_of!(RawRound, eq_first) == 80);
    assert!(std::mem::offset_of!(RawRound, eq_first_len) == 88);
    assert!(std::mem::offset_of!(RawRound, eq_second) == 96);
    assert!(std::mem::offset_of!(RawRound, eq_second_len) == 104);
    assert!(std::mem::offset_of!(RawRound, sources) == 112);
    assert!(std::mem::offset_of!(RawRound, sources_len) == 120);
    assert!(std::mem::offset_of!(RawRound, source_records) == 128);
    assert!(std::mem::offset_of!(RawRound, source_records_len) == 136);
    assert!(std::mem::offset_of!(RawRound, lane_offsets) == 144);
    assert!(std::mem::offset_of!(RawRound, lane_offsets_len) == 152);
    assert!(std::mem::offset_of!(RawRound, references) == 160);
    assert!(std::mem::offset_of!(RawRound, references_len) == 168);
    assert!(std::mem::offset_of!(RawRound, additional_domain_len) == 176);
    assert!(std::mem::offset_of!(RawRound, additional_live_len) == 184);
    assert!(std::mem::offset_of!(RawRound, binary_batching) == 192);
    assert!(std::mem::offset_of!(RawRound, additional_pairs) == 208);
    assert!(std::mem::offset_of!(RawRound, additional_pairs_len) == 216);
    assert!(std::mem::size_of::<Message>() == 160 && std::mem::align_of::<Message>() == 8);
    assert!(std::mem::offset_of!(Message, ordinary) == 0);
    assert!(std::mem::offset_of!(Message, additional) == 96);
};

fn to_u64(x: usize) -> Result<u64> {
    u64::try_from(x).map_err(|_| Error::Admission("length conversion"))
}
fn require(x: bool, reason: &'static str) -> Result<()> {
    if x {
        Ok(())
    } else {
        Err(Error::Admission(reason))
    }
}
impl Round<'_> {
    fn checked(
        &self,
        lanes: usize,
        coefficients: usize,
        initial_domain: usize,
        entry: bool,
    ) -> Result<RawRound> {
        let divisor = if entry { 4 } else { 2 };
        require(
            self.input_coefficients == coefficients
                && coefficients >= divisor * 2
                && coefficients.is_power_of_two(),
            "round shape",
        )?;
        if !entry {
            require(self.r1 == Element::default(), "advance r1 must be zero")?;
        }
        let m = coefficients / divisor;
        let live = lanes
            .checked_mul(m)
            .ok_or(Error::Admission("live overflow"))?;
        require(
            initial_domain.is_power_of_two()
                && initial_domain
                    >= lanes
                        .checked_mul(coefficients)
                        .ok_or(Error::Admission("domain overflow"))?
                && initial_domain <= MAX_N,
            "stored domain",
        )?;
        let domain = initial_domain / divisor;
        require(
            self.alpha.len() == m && self.lane_weights.len() == lanes,
            "alpha/lane extent",
        )?;
        let pairs = live / 2;
        require(
            self.eq_first.len().is_power_of_two() && self.eq_first.len() <= pairs,
            "first equality extent",
        )?;
        require(
            self.eq_second.len() == pairs.div_ceil(self.eq_first.len()),
            "second equality extent",
        )?;
        require(
            self.sources.len() <= MAX_N
                && self.source_records.len() <= MAX_DESCRIPTORS
                && self.references.len() <= MAX_DESCRIPTORS
                && self.additional_pairs.len() <= MAX_DESCRIPTORS,
            "descriptor capacity",
        )?;
        require(
            self.lane_offsets.len()
                == lanes
                    .checked_add(1)
                    .ok_or(Error::Admission("lane overflow"))?,
            "CSR extent",
        )?;
        let refs_len = to_u64(self.references.len())?;
        require(
            self.lane_offsets.first() == Some(&0) && self.lane_offsets.last() == Some(&refs_len),
            "CSR endpoints",
        )?;
        require(
            self.lane_offsets
                .windows(2)
                .all(|p| p[0] <= p[1] && p[1] <= refs_len),
            "CSR order",
        )?;
        let mut next = 0u64;
        for source in self.source_records {
            require(source.offset == next, "source offsets")?;
            next = next
                .checked_add(
                    source
                        .lanes
                        .checked_mul(to_u64(m)?)
                        .ok_or(Error::Admission("source overflow"))?,
                )
                .ok_or(Error::Admission("source overflow"))?;
            require(next <= to_u64(self.sources.len())?, "source span")?;
        }
        require(next == to_u64(self.sources.len())?, "source coverage")?;
        for reference in self.references {
            let idx = usize::try_from(reference.source)
                .map_err(|_| Error::Admission("source conversion"))?;
            let source = self
                .source_records
                .get(idx)
                .ok_or(Error::Admission("source index"))?;
            require(reference.lane < source.lanes, "source lane")?;
        }
        require(
            self.additional_live_len == live && self.additional_domain_len == domain,
            "additional domain",
        )?;
        let mut previous = None;
        for pair in self.additional_pairs {
            require(
                pair.parent < to_u64(domain / 2)? && previous.is_none_or(|old| old < pair.parent),
                "additional order/index",
            )?;
            previous = Some(pair.parent);
        }
        Ok(RawRound {
            input_coefficients: to_u64(coefficients)?,
            skip_linear: u64::from(self.skip_linear),
            r0: self.r0,
            r1: self.r1,
            alpha: self.alpha.as_ptr(),
            alpha_len: to_u64(self.alpha.len())?,
            lane_weights: self.lane_weights.as_ptr(),
            lane_weights_len: to_u64(self.lane_weights.len())?,
            eq_first: self.eq_first.as_ptr(),
            eq_first_len: to_u64(self.eq_first.len())?,
            eq_second: self.eq_second.as_ptr(),
            eq_second_len: to_u64(self.eq_second.len())?,
            sources: self.sources.as_ptr(),
            sources_len: to_u64(self.sources.len())?,
            source_records: self.source_records.as_ptr(),
            source_records_len: to_u64(self.source_records.len())?,
            lane_offsets: self.lane_offsets.as_ptr(),
            lane_offsets_len: to_u64(self.lane_offsets.len())?,
            references: self.references.as_ptr(),
            references_len: to_u64(self.references.len())?,
            additional_domain_len: to_u64(domain)?,
            additional_live_len: to_u64(live)?,
            binary_batching: self.binary_batching,
            additional_pairs: self.additional_pairs.as_ptr(),
            additional_pairs_len: to_u64(self.additional_pairs.len())?,
        })
    }
}
/// Native thread affinity is mirrored by Rc phantom ownership: neither Send nor
/// Sync. Drop always destroys on the creating thread under safe Rust use.
pub struct Owner {
    ptr: Option<NonNull<c_void>>,
    lanes: usize,
    coefficients: usize,
    domain: usize,
    compact: bool,
    _local: PhantomData<Rc<()>>,
}
impl Owner {
    /// Pure native admission plus device/pipeline creation, all before the caller
    /// starts the Stage2 transcript. Creation errors release partial native state.
    pub fn create(config: Config, compact: &[i8]) -> Result<Self> {
        config.validate_compact(compact)?;
        let count = to_u64(compact.len())?;
        // SAFETY: repr(C) config and borrowed initialized bytes remain live for calls.
        let status = unsafe { akita_stage2_admit(&config.raw, compact.as_ptr(), count) };
        if status != 0 {
            return Err(Error::Native(status));
        }
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { akita_stage2_create(&config.raw, compact.as_ptr(), count, &mut ptr) };
        if status != 0 {
            if !ptr.is_null() {
                unsafe { akita_stage2_destroy(ptr) };
            }
            return Err(Error::Native(status));
        }
        let ptr = NonNull::new(ptr).ok_or(Error::Native(6))?;
        Ok(Self {
            ptr: Some(ptr),
            lanes: config.raw.lanes as usize,
            coefficients: config.raw.coefficients as usize,
            domain: config.raw.initial_domain_len as usize,
            compact: true,
            _local: PhantomData,
        })
    }
    // Safe Rust cannot move this owner across threads. Raw ABI wrong-thread
    // destruction deliberately refuses to free; ignoring its status here relies
    // on the private pointer and !Send/!Sync invariant, not a recovery path.
    fn close(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            unsafe { akita_stage2_destroy(ptr.as_ptr()) };
        }
    }
    pub fn compact_entry(&mut self, round: &Round<'_>) -> Result<Message> {
        self.step(round, true)
    }
    pub fn advance(&mut self, round: &Round<'_>) -> Result<Message> {
        self.step(round, false)
    }
    fn step(&mut self, round: &Round<'_>, entry: bool) -> Result<Message> {
        let ptr = self.ptr.ok_or(Error::Closed)?;
        require(self.compact == entry, "operation phase")?;
        let raw = round.checked(self.lanes, self.coefficients, self.domain, entry)?;
        let mut out = Message::default();
        // SAFETY: checked spans borrow initialized repr(C) slices for this synchronous
        // call. Private handle belongs to this thread; output is disjoint stack storage.
        let status = unsafe {
            if entry {
                akita_stage2_compact_entry(ptr.as_ptr(), &raw, &mut out)
            } else {
                akita_stage2_advance(ptr.as_ptr(), &raw, &mut out)
            }
        };
        if status != 0 {
            self.close();
            return Err(Error::Native(status));
        }
        if out
            .ordinary
            .iter()
            .chain(out.additional.iter())
            .any(|x| !x.canonical())
        {
            self.close();
            return Err(Error::NoncanonicalOutput);
        }
        self.coefficients /= if entry { 4 } else { 2 };
        self.domain /= if entry { 4 } else { 2 };
        self.compact = false;
        Ok(out)
    }
    /// Consuming readback initializes exactly L*C elements; no zero-copy type casts.
    pub fn finish(mut self) -> Result<Vec<Element>> {
        require(!self.compact, "entry not completed")?;
        let ptr = self.ptr.ok_or(Error::Closed)?;
        let n = self
            .lanes
            .checked_mul(self.coefficients)
            .ok_or(Error::Admission("finish overflow"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| Error::Allocation)?;
        out.resize(n, Element::default());
        let status = unsafe { akita_stage2_finish(ptr.as_ptr(), out.as_mut_ptr(), to_u64(n)?) };
        if status != 0 {
            return Err(Error::Native(status));
        }
        if out.iter().any(|x| !x.canonical()) {
            return Err(Error::NoncanonicalOutput);
        }
        self.close();
        Ok(out)
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
use tests::mock::{
    akita_stage2_admit, akita_stage2_advance, akita_stage2_compact_entry, akita_stage2_create,
    akita_stage2_destroy, akita_stage2_finish,
};
#[cfg(test)]
mod tests;
