//! Validated one-hot source layouts and device encodings.

use akita_cpu_backend::{OneHotIndex, OneHotSource};
use akita_error::checked;
use jolt_metal::runtime::DeviceBuffer;

use super::{shape_error, NO_HOT};
use crate::{AkitaMetal, AkitaMetalError};

/// One uploaded source; logical fields can be shorter than backing chunks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceLayout {
    pub(crate) hot_offset: usize,
    pub(crate) chunks: usize,
    pub(crate) fields: usize,
    pub(crate) log_chunk: u32,
    pub(crate) num_vars: usize,
}

/// One u32 per chunk, with u32::MAX denoting an absent entry.
pub struct DeviceOneHotSources {
    pub(crate) hot: DeviceBuffer<u32>,
    pub(crate) sources: Vec<SourceLayout>,
}

impl DeviceOneHotSources {
    /// Validates and uploads sources, preserving logical and backing lengths.
    pub fn new<I: OneHotIndex>(
        metal: &AkitaMetal,
        sources: &[OneHotSource<'_, I>],
    ) -> Result<Self, AkitaMetalError> {
        let (hot, sources) = stage_sources::<_, _, false>(sources, std::convert::identity)?;
        Ok(Self {
            hot: DeviceBuffer::from_slice(metal.device(), &hot)?,
            sources,
        })
    }
}

/// K16 sources with one byte per chunk: 0..15 are present, 255 is absent.
///
/// This is an alternative encoding of the same source, not a different
/// polynomial. Logical source bounds remain separate from backing storage.
/// Absent suffixes are omitted from device backing; present zero digits stay.
pub struct DeviceK16Sources {
    pub(crate) hot: DeviceBuffer<u8>,
    pub(crate) sources: Vec<SourceLayout>,
}

impl DeviceK16Sources {
    /// Rejects non-K16 sources before staging or allocating a device buffer.
    pub fn new<I: OneHotIndex>(
        metal: &AkitaMetal,
        sources: &[OneHotSource<'_, I>],
    ) -> Result<Self, AkitaMetalError> {
        if sources.iter().any(|source| source.chunk_size != 16) {
            return Err(shape_error("packed K16 sources require chunk size 16"));
        }
        // stage_sources validates present entries against 16. The only other
        // value is NO_HOT, whose low byte is the absent sentinel 255.
        let (hot, sources) = stage_sources::<_, _, true>(sources, |value| value as u8)?;
        Ok(Self {
            hot: DeviceBuffer::from_slice(metal.device(), &hot)?,
            sources,
        })
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "before each push len < the checked total; every source starts at or before the final len"
)]
fn stage_sources<I: OneHotIndex, T, const TRIM_ABSENT: bool>(
    sources: &[OneHotSource<'_, I>],
    encode: impl Fn(u32) -> T,
) -> Result<(Vec<T>, Vec<SourceLayout>), AkitaMetalError> {
    let total = checked::sum(sources.iter().map(|source| source.indices.len()))
        .ok_or_else(|| shape_error("one-hot chunk count overflows usize"))?;
    let mut hot = Vec::new();
    hot.try_reserve_exact(total)
        .map_err(|_| shape_error(format!("cannot stage {total} one-hot chunks")))?;
    let mut layouts = Vec::with_capacity(sources.len());
    for source in sources {
        let chunk_size = source.chunk_size;
        if !chunk_size.is_power_of_two() || u32::try_from(chunk_size).is_err() {
            return Err(shape_error(format!(
                "one-hot chunk size {chunk_size} is not a power of two below 2^32"
            )));
        }
        let fields = u32::try_from(source.num_vars)
            .ok()
            .and_then(|num_vars| 1usize.checked_shl(num_vars))
            .ok_or_else(|| {
                shape_error(format!(
                    "2^{} one-hot entries overflow the host address space",
                    source.num_vars
                ))
            })?;
        let hot_offset = hot.len();
        let mut last_present_end = hot_offset;
        for index in source.indices {
            hot.push(encode(match index {
                None => NO_HOT,
                // A chunk size below 2^32 keeps every valid index below
                // NO_HOT.
                Some(index) => {
                    let index = u32::try_from(index.as_usize())
                        .ok()
                        .filter(|&index| (index as usize) < chunk_size)
                        .ok_or_else(|| {
                            shape_error(format!(
                                "hot index {} is outside a chunk of {chunk_size}",
                                index.as_usize()
                            ))
                        })?;
                    if TRIM_ABSENT {
                        last_present_end = hot.len() + 1;
                    }
                    index
                }
            }));
        }
        if TRIM_ABSENT {
            hot.truncate(last_present_end);
        }
        layouts.push(SourceLayout {
            hot_offset,
            chunks: hot.len() - hot_offset,
            fields,
            log_chunk: chunk_size.trailing_zeros(),
            num_vars: source.num_vars,
        });
    }
    Ok((hot, layouts))
}
