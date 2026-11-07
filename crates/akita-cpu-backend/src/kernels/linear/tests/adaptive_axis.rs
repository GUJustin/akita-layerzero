use super::*;
use akita_algebra::ntt::tables::q128_primes;
use jolt_field::Prime128OffsetA7F7;

#[test]
fn adaptive_matvec_axis_matches_sparse_convolution_with_partial_blocks() {
    type F = Prime128OffsetA7F7;
    const D: usize = 64;
    // Two column tiles versus eight input blocks. Three-plane alignment also
    // exercises dispatch when the inputs are decomposed from field elements.
    let width = (base_tile_width::<i32, Q128_NUM_PRIMES, D>(1) + 22).div_ceil(3) * 3;
    let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
    let scalars: Vec<F> = (0..width)
        .map(|i| {
            F::from_u128(
                (((i as u128 + 17) * 0x9E37_79B1_85EB_CA87) << 64)
                    | (i as u128 * 0xC2B2_AE3D_27D4_EB4F),
            )
        })
        .collect();
    let matrix = vec![scalars
        .iter()
        .map(|&scalar| {
            let mut polynomial = CyclotomicRing::<F, D>::zero();
            polynomial.coeffs[0] = scalar;
            polynomial.coeffs[D - 1] = -scalar;
            polynomial
        })
        .collect()];
    let prepared = precompute_dense_mat_ntt_with_params(&matrix, &params);
    let matrix_views: Vec<&[_]> = prepared.iter().map(Vec::as_slice).collect();
    let blocks: Vec<Vec<[i8; D]>> = [
        width,
        width - 3,
        width / 2,
        width,
        width,
        width / 3,
        width,
        0,
    ]
    .into_iter()
    .enumerate()
    .map(|(block, len)| {
        (0..len)
            .map(|column| {
                if column % 11 == 0 {
                    [0; D]
                } else {
                    std::array::from_fn(|coefficient| {
                        ((column + 3 * coefficient + block) % 8) as i8 - 4
                    })
                }
            })
            .collect()
    })
    .collect();
    let block_views: Vec<&[[i8; D]]> = blocks.iter().map(Vec::as_slice).collect();
    // Also exercise an odd digit count. Tile boundaries must remain aligned
    // to the three planes that reconstruct each input ring.
    let ring_blocks: Vec<Vec<CyclotomicRing<F, D>>> = blocks
        .iter()
        .map(|block| {
            block
                .chunks(3)
                .map(|planes| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                        F::from_i64(
                            planes
                                .iter()
                                .enumerate()
                                .map(|(digit, plane)| {
                                    i64::from(plane[coefficient]) * (1i64 << (3 * digit))
                                })
                                .sum(),
                        )
                    }))
                })
                .collect()
        })
        .collect();
    let ring_views: Vec<&[_]> = ring_blocks.iter().map(Vec::as_slice).collect();
    let expected: Vec<Vec<CyclotomicRing<F, D>>> = blocks
        .iter()
        .map(|digits| {
            let mut output = CyclotomicRing::<F, D>::zero();
            for (&scalar, digit) in scalars.iter().zip(digits) {
                // Multiply by scalar * (1 - X^(D-1)) modulo X^D + 1.
                // This oracle uses no NTT or CRT reconstruction.
                for (i, &digit) in digit.iter().enumerate() {
                    let product = scalar * F::from_i64(i64::from(digit));
                    output.coeffs[i] += product;
                    if i == 0 {
                        output.coeffs[D - 1] -= product;
                    } else {
                        output.coeffs[i - 1] += product;
                    }
                }
            }
            vec![output]
        })
        .collect();
    let check = || {
        let actual = mat_vec_mul_digits_i8_with_params::<F, _, Q128_NUM_PRIMES, D>(
            &matrix_views,
            &block_views,
            3,
            &params,
        )
        .unwrap();
        assert_eq!(actual, expected);
        let from_rings = mat_vec_mul_i8_with_params::<F, _, Q128_NUM_PRIMES, D>(
            &matrix_views,
            &ring_views,
            3,
            3,
            &params,
        )
        .unwrap();
        assert_eq!(from_rings, expected);
    };
    #[cfg(feature = "parallel")]
    for workers in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(check);
    }
    #[cfg(not(feature = "parallel"))]
    check();
}
