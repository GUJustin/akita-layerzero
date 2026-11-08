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

fn stage_sources<I: OneHotIndex, T: Copy + Send, const TRIM_ABSENT: bool>(
    sources: &[OneHotSource<'_, I>],
    encode: impl Fn(u32) -> T + Sync,
) -> Result<(Vec<T>, Vec<SourceLayout>), AkitaMetalError> {
    checked::sum(sources.iter().map(|source| source.indices.len()))
        .ok_or_else(|| shape_error("one-hot chunk count overflows usize"))?;
    let mut total = 0usize;
    let mut layouts = Vec::with_capacity(sources.len());
    for source in sources {
        let chunk_size = source.chunk_size;
        if !chunk_size.is_power_of_two() || u32::try_from(chunk_size).is_err() {
            return Err(shape_error(format!(
                "one-hot chunk size {chunk_size} is not a power of two below 2^32"
            )));
        }
        let fields = checked::pow2(source.num_vars).ok_or_else(|| {
            shape_error(format!(
                "2^{} one-hot entries overflow the host address space",
                source.num_vars
            ))
        })?;
        // Only None entries may be omitted. In particular, Some(0) remains
        // present, and a present index beyond the logical domain is validated.
        let chunks = if TRIM_ABSENT {
            source
                .indices
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |last| {
                    // An index into a slice is strictly smaller than its length.
                    last.saturating_add(1)
                })
        } else {
            source.indices.len()
        };
        layouts.push(SourceLayout {
            hot_offset: total,
            chunks,
            fields,
            log_chunk: chunk_size.trailing_zeros(),
            num_vars: source.num_vars,
        });
        total = total
            .checked_add(chunks)
            .ok_or_else(|| shape_error("one-hot chunk count overflows usize"))?;
    }
    let mut hot = Vec::new();
    hot.try_reserve_exact(total)
        .map_err(|_| shape_error(format!("cannot stage {total} one-hot chunks")))?;
    hot.resize(total, encode(NO_HOT));
    let mut rest = hot.as_mut_slice();
    let mut work = Vec::with_capacity(sources.len());
    for (source, layout) in sources.iter().zip(&layouts) {
        let (destination, remaining) = rest.split_at_mut(layout.chunks);
        rest = remaining;
        work.push((source, destination));
    }
    let fill = |(source, destination): (&OneHotSource<'_, I>, &mut [T])| {
        for (index, out) in source.indices.iter().zip(destination) {
            *out = encode(match index {
                None => NO_HOT,
                Some(index) => u32::try_from(index.as_usize())
                    .ok()
                    .filter(|&index| (index as usize) < source.chunk_size)
                    .ok_or_else(|| {
                        shape_error(format!(
                            "hot index {} is outside a chunk of {}",
                            index.as_usize(),
                            source.chunk_size
                        ))
                    })?,
            });
        }
        Ok::<_, AkitaMetalError>(())
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        if total >= 65_536 && work.len() > 1 {
            work.into_par_iter().try_for_each(fill)?;
        } else {
            work.into_iter().try_for_each(fill)?;
        }
    }
    #[cfg(not(feature = "parallel"))]
    work.into_iter().try_for_each(fill)?;
    Ok((hot, layouts))
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
