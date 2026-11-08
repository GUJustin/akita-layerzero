//! Regression oracle for the former serial source encoding.
#![allow(clippy::arithmetic_side_effects, clippy::indexing_slicing)]
use super::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn layouts_equal(a: &[SourceLayout], b: &[SourceLayout]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(
            (a.hot_offset, a.chunks, a.fields, a.log_chunk, a.num_vars),
            (b.hot_offset, b.chunks, b.fields, b.log_chunk, b.num_vars)
        );
    }
}

#[test]
fn source_packing_preserves_bytes_and_layouts() {
    let mut rng = StdRng::seed_from_u64(4721);
    for length in [0, 1, 31, 32, 33, 1023, 16384, 65539] {
        let mut columns: Vec<Vec<Option<u8>>> = (0..9)
            .map(|c| {
                (0..length + c)
                    .map(|i| {
                        if i % 5 == 0 {
                            None
                        } else {
                            Some(rng.gen_range(0..16))
                        }
                    })
                    .collect()
            })
            .collect();
        columns[0].fill(None);
        columns[1].fill(Some(0));
        for (i, column) in columns.iter_mut().enumerate() {
            column.extend(std::iter::repeat_n(None, i * 31));
        }
        let sources: Vec<_> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| OneHotSource {
                indices: c,
                chunk_size: 16,
                num_vars: i + 4,
            })
            .collect();
        let (expected, layouts) = serial_reference::<_, _, true>(&sources, |v| v as u8).unwrap();
        let (actual, actual_layouts) = stage_sources::<_, _, true>(&sources, |v| v as u8).unwrap();
        assert_eq!(actual, expected);
        layouts_equal(&layouts, &actual_layouts);
        let (expected, layouts) = serial_reference::<_, _, false>(&sources, |v| v).unwrap();
        let (actual, actual_layouts) = stage_sources::<_, _, false>(&sources, |v| v).unwrap();
        assert_eq!(actual, expected);
        layouts_equal(&layouts, &actual_layouts);
    }
}

#[test]
fn source_packing_rejects_invalid_backing_and_shapes() {
    for value in [16usize, 256, usize::MAX] {
        // The invalid entry lies beyond the logical domain and before an
        // absent suffix; it must still be rejected by both encodings.
        let mut column = vec![Some(0); 100_000];
        column.push(Some(value));
        column.extend([None; 17]);
        let sources = [OneHotSource {
            indices: &column,
            chunk_size: 16,
            num_vars: 4,
        }; 2];
        assert!(stage_sources::<_, _, true>(&sources, |v| v as u8).is_err());
        assert!(stage_sources::<_, _, false>(&sources, |v| v).is_err());
    }
    for (chunk_size, num_vars) in [(0, 4), (3, 4), (16, usize::BITS as usize), (16, usize::MAX)] {
        let sources = [OneHotSource::<u8> {
            indices: &[],
            chunk_size,
            num_vars,
        }];
        assert!(stage_sources::<_, _, true>(&sources, |v| v as u8).is_err());
    }
    let empty: [OneHotSource<'_, u8>; 0] = [];
    assert!(stage_sources::<_, _, true>(&empty, |v| v as u8)
        .unwrap()
        .0
        .is_empty());
}

#[test]
#[ignore = "manual host staging benchmark; run without other compute jobs"]
fn source_packing_timing() {
    use std::{hint::black_box, time::Instant};
    let mut rng = StdRng::seed_from_u64(9081);
    for length in [16usize, 1024, 65536, 1 << 20] {
        let columns: Vec<Vec<Option<u8>>> = (0..45)
            .map(|c| {
                let mut v: Vec<_> = (0..length)
                    .map(|_| {
                        if rng.gen_bool(0.55) {
                            Some(rng.gen_range(0..16))
                        } else {
                            None
                        }
                    })
                    .collect();
                if c % 5 == 0 {
                    v[length / 2..].fill(None);
                }
                v
            })
            .collect();
        let sources: Vec<_> = columns
            .iter()
            .map(|c| OneHotSource {
                indices: c,
                chunk_size: 16,
                num_vars: length.ilog2() as usize + 4,
            })
            .collect();
        // Warm both implementations and verify their exact outputs first.
        let expected = serial_reference::<_, _, true>(&sources, |v| v as u8).unwrap();
        let actual = stage_sources::<_, _, true>(&sources, |v| v as u8).unwrap();
        assert_eq!(expected.0, actual.0);
        layouts_equal(&expected.1, &actual.1);
        drop((expected, actual));
        for trial in 0..6 {
            let mut times = [0.0; 2];
            for mode in [trial % 2, 1 - trial % 2] {
                let t = Instant::now();
                let result = if mode == 0 {
                    serial_reference::<_, _, true>(&sources, |v| v as u8).unwrap()
                } else {
                    stage_sources::<_, _, true>(&sources, |v| v as u8).unwrap()
                };
                times[mode] = t.elapsed().as_secs_f64();
                black_box(result);
            }
            eprintln!(
                "PACKING length={length} columns=45 trial={trial} old_s={} new_s={} parallel={}",
                times[0],
                times[1],
                cfg!(feature = "parallel")
            );
        }
    }
}

#[expect(
    clippy::arithmetic_side_effects,
    reason = "before each push len < the checked total; every source starts at or before the final len"
)]
fn serial_reference<I: OneHotIndex, T, const TRIM_ABSENT: bool>(
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
