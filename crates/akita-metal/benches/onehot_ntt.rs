//! K16/Fp128 inner commitments: direct sweep vs bounded limb NTT.
//!
//! Synthetic sources contain alternating present/absent chunks. Every output
//! is compared before timing. Measurements reuse resident keys and sources;
//! key preparation and source uploads are reported separately, and are not
//! included in resident timings. These are kernel/API benchmarks, not proofs.

#[cfg(target_os = "macos")]
fn main() {
    use std::time::{Duration, Instant};

    use akita_cpu_backend::OneHotSource;
    use akita_metal::onehot::{
        DeviceFlatMatrix, DeviceK16Sources, DeviceOneHotSources, OneHotCommitShape, OneHotSchedule,
        OneHotWorkspace,
    };
    use akita_metal::onehot_ntt::{DeviceOneHotNttKey, OneHotNttWorkspace};
    use akita_metal::AkitaMetal;
    use criterion::{BenchmarkId, Criterion};
    use jolt_field::{Field, Prime128OffsetA7F7 as F};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let metal = AkitaMetal::new().expect("Apple GPU");
    let mut c = Criterion::default().configure_from_args();
    for (positions, blocks, columns) in [(512usize, 256usize, 45usize), (8192, 16, 45)] {
        let name = format!("p{positions}_b{blocks}_c{columns}");
        let mut rng = StdRng::seed_from_u64(123);
        let a: Vec<F> = (0..positions * 512).map(|_| F::random(&mut rng)).collect();
        let indices: Vec<Vec<Option<u8>>> = (0..columns)
            .map(|column| {
                (0..positions * blocks * 32)
                    .map(|i| ((i + column) % 2 == 0).then(|| rng.gen_range(0..16)))
                    .collect()
            })
            .collect();
        let sources: Vec<_> = indices
            .iter()
            .map(|indices| OneHotSource {
                indices,
                chunk_size: 16,
                num_vars: (positions * blocks * 512).ilog2() as usize,
            })
            .collect();
        let t = Instant::now();
        let key = DeviceOneHotNttKey::new(&metal, &a).expect("prepare key");
        let prepare = t.elapsed();
        let raw_key = DeviceFlatMatrix::new(&metal, &a).expect("upload key");
        let t = Instant::now();
        let packed = DeviceK16Sources::new(&metal, &sources).expect("packed sources");
        let packed_upload = t.elapsed();
        let t = Instant::now();
        let wide = DeviceOneHotSources::new(&metal, &sources).expect("wide sources");
        let wide_upload = t.elapsed();
        let shape = OneHotCommitShape {
            n_a: 1,
            active_a_cols: positions,
            num_digits_inner: 1,
        };
        let schedule = OneHotSchedule::new::<F, 512>(&metal, &wide, shape).expect("schedule");
        let mut direct = OneHotWorkspace::<F>::new(&metal);
        let mut ntt = OneHotNttWorkspace::new(&metal);
        let expected = direct
            .commit::<512>(&raw_key, &wide, shape, schedule)
            .expect("direct")
            .0
            .read()
            .expect("read")
            .to_vec();
        assert_eq!(
            ntt.commit(&key, &packed, shape)
                .expect("NTT")
                .0
                .read()
                .expect("read"),
            expected
        );
        drop(expected);
        eprintln!("{name}: key preparation {prepare:?}; packed upload {packed_upload:?}; wide upload {wide_upload:?}; direct {schedule:?}");
        let mut group = c.benchmark_group("onehot_k16");
        group.sample_size(10);
        for mode in ["direct", "ntt"] {
            let mut run = || {
                if mode == "direct" {
                    direct
                        .commit::<512>(&raw_key, &wide, shape, schedule)
                        .expect("direct")
                        .1
                } else {
                    ntt.commit(&key, &packed, shape).expect("NTT").1
                }
            };
            group.bench_function(BenchmarkId::new(format!("{mode}_gpu"), &name), |b| {
                b.iter_custom(|iters| (0..iters).map(|_| run()).sum::<Duration>());
            });
            group.bench_function(BenchmarkId::new(format!("{mode}_wall"), &name), |b| {
                b.iter(&mut run);
            });
        }
        group.finish();
    }
    c.final_summary();
}

#[cfg(not(target_os = "macos"))]
fn main() {}
