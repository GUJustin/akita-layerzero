//! Limb-NTT commitments against the CPU sweep and independent ring shifts.
#![cfg(target_os = "macos")]

mod support;

use akita_algebra::CyclotomicRing;
use akita_cpu_backend::benchmark_support::column_sweep_ajtai_onehot_multi;
use akita_cpu_backend::OneHotSource;
use akita_metal::onehot::{DeviceK16Sources, OneHotCommitShape};
use akita_metal::onehot_ntt::{DeviceOneHotNttKey, OneHotNttWorkspace};
use akita_params::FlatMatrix;
use jolt_field::{Field, One, Prime128OffsetA7F7 as F, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

const D: usize = 512;

fn matrix(len: usize, rng: &mut StdRng) -> Vec<F> {
    (0..len)
        .map(|i| match i % 31 {
            0 => F::zero(),
            1 => F::one(),
            2 => -F::one(),
            _ => F::random(rng),
        })
        .collect()
}

fn model(a: &[F], sources: &[OneHotSource<'_, u8>], shape: OneHotCommitShape) -> Vec<F> {
    let positions = shape.active_a_cols / shape.num_digits_inner;
    let mut out = Vec::new();
    for source in sources {
        let fields = 1usize << source.num_vars;
        let blocks = (fields / D).div_ceil(positions);
        let mut rows = vec![CyclotomicRing::<F, D>::zero(); blocks * shape.n_a];
        for (chunk, hot) in source.indices.iter().enumerate() {
            let Some(hot) = hot else { continue };
            let field = chunk * 16 + usize::from(*hot);
            if field >= fields {
                continue;
            }
            let ring = field / D;
            let block = ring / positions;
            let position = ring % positions;
            for row in 0..shape.n_a {
                let start = (row * shape.active_a_cols + position * shape.num_digits_inner) * D;
                let term = CyclotomicRing::<F, D>::from_slice(&a[start..start + D])
                    .negacyclic_shift(field % D);
                rows[block * shape.n_a + row] += term;
            }
        }
        out.extend(
            rows.iter()
                .flat_map(|ring| ring.coefficients().iter().copied()),
        );
    }
    out
}

#[test]
fn limb_commitments_match_cpu_across_rows_digits_and_sources() {
    let gpu = support::gpu();
    let metal = &gpu.metal;
    let mut rng = StdRng::seed_from_u64(901);
    let mut workspace = OneHotNttWorkspace::new(metal);
    for (positions, rows, digits) in [(1usize, 1, 1), (4, 3, 3), (256, 1, 1), (512, 2, 3)] {
        let shape = OneHotCommitShape {
            n_a: rows,
            active_a_cols: positions * digits,
            num_digits_inner: digits,
        };
        let a = matrix(rows * shape.active_a_cols * D, &mut rng);
        let key = DeviceOneHotNttKey::new(metal, &a).expect("prepare key");
        let vars = positions.ilog2() as usize + D.ilog2() as usize + 1;
        let indices: Vec<_> = (0..(1usize << vars) / 16)
            .map(|i| match i % 7 {
                0 => None,
                1 => Some(0),
                2 => Some(15),
                _ => Some(rng.gen_range(0..16)),
            })
            .collect();
        let mut first = vec![Some(0u8); 32];
        first.resize(64, None); // A present zero at the end must survive trimming.
        let empty = vec![None; 32];
        let sources = [
            OneHotSource {
                indices: &indices,
                chunk_size: 16,
                num_vars: vars,
            },
            OneHotSource {
                indices: &first,
                chunk_size: 16,
                num_vars: 9,
            },
            OneHotSource {
                indices: &empty,
                chunk_size: 16,
                num_vars: 9,
            },
            // Exercise both orders: a live source next to an empty one,
            // and an empty source followed by a live one.
            OneHotSource {
                indices: &empty,
                chunk_size: 16,
                num_vars: 9,
            },
            OneHotSource {
                indices: &first,
                chunk_size: 16,
                num_vars: 9,
            },
        ];
        let expected = model(&a, &sources, shape);
        let flat = FlatMatrix::from_flat_data(a.clone());
        let view = flat
            .ring_view::<D>(rows, shape.active_a_cols)
            .expect("matrix view");
        let cpu: Vec<F> = column_sweep_ajtai_onehot_multi::<F, D, u8>(
            &view,
            &sources,
            rows,
            shape.active_a_cols,
            digits,
        )
        .expect("CPU sweep")
        .into_iter()
        .flatten()
        .flatten()
        .flat_map(|ring| *ring.coefficients())
        .collect();
        assert_eq!(cpu, expected, "independent shifts vs CPU sweep");
        let device = DeviceK16Sources::new(metal, &sources).expect("packed sources");
        for _ in 0..2 {
            let (output, _) = workspace.commit(&key, &device, shape).expect("commit");
            assert_eq!(
                output.read().expect("canonical output"),
                expected.as_slice()
            );
        }
    }
}

#[test]
fn one_prepared_key_supports_shorter_sources_and_views() {
    let gpu = support::gpu();
    let metal = &gpu.metal;
    let mut rng = StdRng::seed_from_u64(902);
    let a = matrix(512 * D, &mut rng);
    let key = DeviceOneHotNttKey::new(metal, &a).expect("key independent of trace length");
    let mut workspace = OneHotNttWorkspace::new(metal);
    for (positions, rows, digits, vars, backing) in [
        (512, 1, 1, 9, 64),        // backing extends past the logical source
        (4, 2, 3, 13, 13),         // short backing and a different matrix view
        (512, 1, 1, 19, 32 * 512), // two blocks, second one absent
        (1, 1, 1, 9, 32),
    ] {
        let shape = OneHotCommitShape {
            n_a: rows,
            active_a_cols: positions * digits,
            num_digits_inner: digits,
        };
        let indices: Vec<_> = (0..backing)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some((i % 16) as u8)
                }
            })
            .collect();
        let sources = [OneHotSource {
            indices: &indices,
            chunk_size: 16,
            num_vars: vars,
        }];
        let expected = model(&a, &sources, shape);
        let device = DeviceK16Sources::new(metal, &sources).expect("source");
        let (out, _) = workspace.commit(&key, &device, shape).expect("reused key");
        assert_eq!(out.read().expect("read"), expected.as_slice());
    }
    // Enough output rings select the no-scratch path, across three
    // reconstruction chunks. Only three entries are present, so the
    // independent oracle remains cheap despite the large logical source.
    let mut indices = vec![None; 512 * 32];
    for (chunk, digit) in [(0, 0), (252 * 32 + 1, 15), (511 * 32 + 31, 7)] {
        indices[chunk] = Some(digit);
    }
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 512,
        num_digits_inner: 1,
    };
    for clear in [false, true] {
        if clear {
            indices.fill(None);
        }
        let sources = [OneHotSource {
            indices: &indices,
            chunk_size: 16,
            num_vars: 31,
        }];
        let expected = model(&a, &sources, shape);
        let device = DeviceK16Sources::new(metal, &sources).expect("source");
        let (out, _) = workspace
            .commit(&key, &device, shape)
            .expect("direct accumulation");
        assert_eq!(out.read().expect("read"), expected.as_slice());
    }
}

#[test]
fn maximal_limbs_cross_reconstruction_boundaries_without_wraparound() {
    let gpu = support::gpu();
    let metal = &gpu.metal;
    let a = vec![-F::one(); 512 * D];
    let key = DeviceOneHotNttKey::new(metal, &a).expect("maximal-limb key");
    let mut workspace = OneHotNttWorkspace::new(metal);
    // Dense sources maximize the integer convolution bound. Last-position
    // digits exercise negative wrap; 252/253 straddle reconstruction chunks.
    // Absent tails exercise skipping later partials without dropping Some(0).
    let backing: Vec<Vec<Option<u8>>> = [251, 252, 253, 504, 512]
        .into_iter()
        .flat_map(|positions| {
            [0, 15].map(|digit| {
                let mut indices = vec![Some(digit); positions * 32];
                indices.resize(512 * 32, None);
                indices
            })
        })
        .collect();
    let sources: Vec<_> = backing
        .iter()
        .map(|indices| OneHotSource {
            indices,
            chunk_size: 16,
            num_vars: 18,
        })
        .collect();
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 512,
        num_digits_inner: 1,
    };
    let expected = model(&a, &sources, shape);
    let device = DeviceK16Sources::new(metal, &sources).expect("dense sources");
    let (out, _) = workspace
        .commit(&key, &device, shape)
        .expect("dense commit");
    assert_eq!(out.read().expect("read"), expected.as_slice());
}

#[test]
fn rejects_invalid_shapes_and_keeps_workspace_reusable() {
    let gpu = support::gpu();
    let metal = &gpu.metal;
    let mut rng = StdRng::seed_from_u64(903);
    let a = matrix(D, &mut rng);
    assert!(DeviceOneHotNttKey::new(metal, &a[..D - 1]).is_err());
    let key = DeviceOneHotNttKey::new(metal, &a).expect("key");
    for (chunk_size, index) in [(8, 0u8), (16, 16), (16, 255)] {
        let indices = [Some(index)];
        assert!(DeviceK16Sources::new(
            metal,
            &[OneHotSource {
                indices: &indices,
                chunk_size,
                num_vars: 9,
            }]
        )
        .is_err());
    }
    let indices = [Some(0u8)];
    assert!(DeviceK16Sources::new(
        metal,
        &[OneHotSource {
            indices: &indices,
            chunk_size: 16,
            num_vars: usize::MAX,
        }]
    )
    .is_err());
    let sources = [OneHotSource {
        indices: &indices,
        chunk_size: 16,
        num_vars: 9,
    }];
    let device = DeviceK16Sources::new(metal, &sources).expect("source");
    let mut workspace = OneHotNttWorkspace::new(metal);
    for shape in [
        OneHotCommitShape {
            n_a: 1,
            active_a_cols: 1,
            num_digits_inner: 0,
        },
        OneHotCommitShape {
            n_a: 1,
            active_a_cols: 3,
            num_digits_inner: 1,
        },
        OneHotCommitShape {
            n_a: 1,
            active_a_cols: 2,
            num_digits_inner: 1,
        },
        OneHotCommitShape {
            n_a: usize::MAX,
            active_a_cols: 2,
            num_digits_inner: 1,
        },
    ] {
        assert!(workspace.commit(&key, &device, shape).is_err());
    }
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 1,
        num_digits_inner: 1,
    };
    let (out, _) = workspace
        .commit(&key, &device, shape)
        .expect("valid call after rejections");
    assert_eq!(out.read().expect("read"), a.as_slice());
    let empty = DeviceK16Sources::new::<u8>(metal, &[]).expect("empty sources");
    let (out, _) = workspace.commit(&key, &empty, shape).expect("no sources");
    assert!(out.read().expect("read").is_empty());
    let zero_rank = OneHotCommitShape { n_a: 0, ..shape };
    let (out, _) = workspace
        .commit(&key, &device, zero_rank)
        .expect("zero rank");
    assert!(out.read().expect("read").is_empty());
    let wide_empty = OneHotCommitShape {
        n_a: 0,
        active_a_cols: 1 << 31,
        num_digits_inner: 1,
    };
    let (out, _) = workspace
        .commit(&key, &empty, wide_empty)
        .expect("wide empty view");
    assert!(out.read().expect("read").is_empty());
    let short = DeviceK16Sources::new(
        metal,
        &[OneHotSource {
            indices: &indices,
            chunk_size: 16,
            num_vars: 8,
        }],
    )
    .expect("short logical source");
    assert!(workspace.commit(&key, &short, shape).is_err());
}
