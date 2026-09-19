//! Bounded calling-thread diagnostics for the explicit full-PCS correctness feature.
use akita_error::AkitaError;
use std::cell::RefCell;

/// One successfully completed Stage2 invocation, including preselected CPU routes.
#[derive(Clone, Debug)]
pub struct Event {
    /// Zero-based root/suffix fold level.
    pub level: usize,
    /// Balanced digit basis.
    pub basis: usize,
    /// Initial coefficient width per lane.
    pub columns: usize,
    /// Live lane count.
    pub lanes: usize,
    /// Padded witness domain.
    pub domain: usize,
    /// Actual checked WitnessLayout compression layers.
    pub compression_layers: usize,
    /// Production negative-binary support intervals supplied to AdditionalRelationTerms.
    pub negative_binary_intervals: usize,
    /// Static CPU selection reason, or None for resident completion.
    pub declined: Option<&'static str>,
    /// Successful native compact-entry calls.
    pub compact_entries: usize,
    /// Successful retained native coefficient advances.
    pub advances: usize,
    /// Successful exact final exports.
    pub exports: usize,
}
thread_local! { static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) }; }
pub(crate) fn record(event: Event) -> Result<(), AkitaError> {
    EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        if events.len() >= 64 {
            return Err(AkitaError::InvalidInput("Stage2 observer capacity".into()));
        }
        events.push(event);
        Ok(())
    })
}
/// Drain this thread's bounded events between complete proofs.
pub fn take() -> Vec<Event> {
    EVENTS.with(|events| std::mem::take(&mut *events.borrow_mut()))
}
