use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, IntoPyArray, PyReadonlyArray1};

use rayon::prelude::*;
// use core::simd::f64x2;

use crate::accelerants::FloatArray1;
use std::f64::consts::PI;

// todo: make optimized Python version of this: we have to test
// but it literally might no longer be worth the FFI time

const LOG_THRESHOLD: f64 = 1e-14;

/// Compute the unnormalized integral of r^beta over [start, end].
///   beta != -1:  (end^(beta+1) - start^(beta+1)) / (beta+1)
///   beta == -1:  ln(end/start)
#[inline]
fn region_mass(start: f64, end: f64, beta: f64) -> f64 {
    if start >= end {
        return 0.0;
    }
    let bp1 = beta + 1.0;
    if bp1.abs() < LOG_THRESHOLD {
        // beta ≈ -1 case
        (end / start).ln()
    } else {
        (end.powf(bp1) - start.powf(bp1)) / bp1
    }
}

/// Inverse CDF for f(r) ∝ r^beta over [start, end].
/// Given u ∈ [0, 1), returns r such that CDF(r) = u.
///   beta != -1:  r = ((1-u) * start^(beta+1) + u * end^(beta+1)) ^ (1/(beta+1))
///   beta == -1:  r = start * (end/start)^u
#[inline]
fn icdf(u: f64, start: f64, end: f64, beta: f64) -> f64 {
    let bp1 = beta + 1.0;
    
    if bp1.abs() < LOG_THRESHOLD {
        start * (end / start).powf(u)
    } else {
        let start_bp1 = start.powf(bp1);
        let end_bp1 = end.powf(bp1);
        ((1.0 - u) * start_bp1 + u * end_bp1).powf(1.0 / bp1)
    }
}

/// Directly sample from the piecewise power-law distribution using
/// inverse CDF transform. No grid, no PDF array, one powf per sample.
///
/// Arguments match generate_r's convention:
///   start, end       — radial range [start, end]
///   crit_radius      — breakpoint between inner and outer power law
///   index_inner/outer — power-law indices (positive; negated internally)
///   volume_scaling    — if true, weight by r^2 (spherical shell volume)
///   sample_arr        — uniform random draws in [0, 1), one per BH
///
/// Returns: array of radial positions, same length as samples.
#[pyfunction]
pub fn sample_powerlaw_icdf<'py>(
    py: Python<'py>,
    start: f64,
    end: f64,
    crit_radius: f64,
    index_inner: f64,
    index_outer: f64,
    volume_scaling: bool,
    sample_arr: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // Effective exponent: f(r) ∝ r^beta
    // Without volume scaling: f(r) ∝ (r/r_c)^(-alpha) ∝ r^(-alpha)
    // With volume scaling:    f(r) ∝ r^2 * r^(-alpha) = r^(2 - alpha)
    let beta_inner = if volume_scaling { 2.0 - index_inner } else { -index_inner };
    let beta_outer = if volume_scaling { 2.0 - index_outer } else { -index_outer };

    // Clamp crit_radius to [start, end] so both regions are well-defined
    let r_c = crit_radius.clamp(start, end);

    // Unnormalized probability mass of each region
    let w_inner = region_mass(start, r_c, beta_inner);
    let w_outer = region_mass(r_c, end, beta_outer);
    let w_total = w_inner + w_outer;

    if w_total <= 0.0 || !w_total.is_finite() {
        return Err(PyValueError::new_err(
            "[sample_powerlaw_icdf] Total probability mass is zero or non-finite. \
             Check that start < end and indices are valid."
        ));
    }

    let p_inner = w_inner / w_total;

    let sample_slice = sample_arr.as_slice()?;
    let n = sample_slice.len();
    let mut result = Vec::with_capacity(n);

    for &sample in sample_slice {
        let r = if sample < p_inner {
            // Rescale u to [0, 1) within the inner region
            let u_inner = sample / p_inner;
            icdf(u_inner, start, r_c, beta_inner)
        } else {
            // Rescale u to [0, 1) within the outer region
            let u_outer = (sample - p_inner) / (1.0 - p_inner);
            icdf(u_outer, r_c, end, beta_outer)
        };
        result.push(r);
    }

    Ok(result.into_pyarray(py))
}








// /// SIMD kernel: processes one contiguous chunk, called from a rayon task.
// #[inline(always)]
// fn fill_chunk<const VOL: bool>(
//     r_chunk: &mut [f64],
//     y_chunk: &mut [f64],
//     start: f64,
//     step: f64,
//     inv_crit: f64,
//     neg_index: f64,
//     global_offset: usize,
// ) {
//     let len = r_chunk.len();
//     let step_v  = f64x2::splat(step);
//     let inv_v   = f64x2::splat(inv_crit);
//     let exp_v   = f64x2::splat(neg_index);
//     let pi_step = f64x2::splat(PI * step);
//
//     let pairs = len / 2;
//     for c in 0..pairs {
//         let i0 = c * 2;
//         let i1 = i0 + 1;
//         let idx = f64x2::from_array([
//             (i0 + global_offset) as f64,
//             (i1 + global_offset) as f64,
//         ]);
//         let vals = f64x2::splat(start) + idx * step_v;
//         let bases = vals * inv_v;
//         let p = sleef::f64x::pow_u10(bases, exp_v);
//
//         let v = vals.to_array();
//         r_chunk[i0] = v[0];
//         r_chunk[i1] = v[1];
//
//         if VOL {
//             let y = pi_step * vals * vals * p;
//             let ya = y.to_array();
//             y_chunk[i0] = ya[0];
//             y_chunk[i1] = ya[1];
//         } else {
//             let pa = p.to_array();
//             y_chunk[i0] = pa[0];
//             y_chunk[i1] = pa[1];
//         }
//     }
//
//     // scalar tail
//     if len % 2 != 0 {
//         let i = len - 1;
//         let val = start + ((i + global_offset) as f64) * step;
//         r_chunk[i] = val;
//         let base = val * inv_crit;
//         let p = sleef::f64::pow_u10(base, neg_index);
//         y_chunk[i] = if VOL { PI * val * val * step * p } else { p };
//     }
// }
//
// /// Parallel + SIMD fill for one power-law region.
// // fn fill_region<const VOL: bool>(
// //     r_slice: &mut [f64],
// //     y_slice: &mut [f64],
// //     start: f64,
// //     step: f64,
// //     inv_crit: f64,
// //     neg_index: f64,
// //     region_offset: usize,
// // ) {
// //     let len = r_slice.len();
// //     let n_threads = rayon::current_num_threads().max(1);
// //     // Round up to even so SIMD pairs align within each chunk.
// //     let chunk_size = ((len + n_threads - 1) / n_threads + 1) & !1;
// //
// //     r_slice
// //         .chunks_mut(chunk_size)
// //         .zip(y_slice.chunks_mut(chunk_size))
// //         .collect::<Vec<_>>()
// //         .into_par_iter()
// //         .enumerate()
// //         .for_each(|(chunk_idx, (r_chunk, y_chunk))| {
// //             let global_offset = region_offset + chunk_idx * chunk_size;
// //             fill_chunk::<VOL>(
// //                 r_chunk, y_chunk, start, step, inv_crit, neg_index, global_offset,
// //             );
// //         });
// // }
// fn fill_region<const VOL: bool>(
//     r_slice: &mut [f64],
//     y_slice: &mut [f64],
//     start: f64,
//     step: f64,
//     inv_crit: f64,
//     neg_index: f64,
//     region_offset: usize,
// ) {
//     let len = r_slice.len();
//     if len == 0 {
//         return;
//     }
//
//     let n_threads = rayon::current_num_threads().max(1);
//     let chunk_size = ((len + n_threads - 1) / n_threads + 1) & !1;
//     // chunk_size is guaranteed > 0 here since len > 0
//
//     r_slice
//         .chunks_mut(chunk_size)
//         .zip(y_slice.chunks_mut(chunk_size))
//         .collect::<Vec<_>>()
//         .into_par_iter()
//         .enumerate()
//         .for_each(|(chunk_idx, (r_chunk, y_chunk))| {
//             let global_offset = region_offset + chunk_idx * chunk_size;
//             fill_chunk::<VOL>(
//                 r_chunk, y_chunk, start, step, inv_crit, neg_index, global_offset,
//             );
//         });
// }
//
//
// #[pyfunction]
// pub fn generate_r<'py>(
//     py: Python<'py>,
//     start: f64,
//     end: f64,
//     num_points: usize,
//     crit_radius: f64,
//     index_inner: f64,
//     index_outer: f64,
//     volume_scaling: bool,
// ) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {
//     let r_arr = unsafe { PyArray1::new(py, num_points, false) };
//     let r_pdf_arr = unsafe { PyArray1::new(py, num_points, false) };
//
//     let r_slice = unsafe { r_arr.as_slice_mut().unwrap() };
//     let y_slice = unsafe { r_pdf_arr.as_slice_mut().unwrap() };
//
//     if num_points < 2 {
//         return Err(PyValueError::new_err("[generate_r] num_points must be >= 2"));
//     }
//
//     let step = (end - start) / ((num_points - 1) as f64);
//     let neg_inner = -index_inner;
//     let neg_outer = -index_outer;
//     let inv_crit = 1.0 / crit_radius;
//
//     let split_idx = ((crit_radius - start) / step).ceil() as usize;
//     let split_idx = split_idx.min(num_points);
//
//     if volume_scaling {
//         let (r_in, r_out) = r_slice.split_at_mut(split_idx);
//         let (y_in, y_out) = y_slice.split_at_mut(split_idx);
//         fill_region::<true>(r_in, y_in, start, step, inv_crit, neg_inner, 0);
//         fill_region::<true>(r_out, y_out, start, step, inv_crit, neg_outer, split_idx);
//     } else {
//         let (r_in, r_out) = r_slice.split_at_mut(split_idx);
//         let (y_in, y_out) = y_slice.split_at_mut(split_idx);
//         fill_region::<false>(r_in, y_in, start, step, inv_crit, neg_inner, 0);
//         fill_region::<false>(r_out, y_out, start, step, inv_crit, neg_outer, split_idx);
//     }
//
//     let y_sum: f64 = y_slice.iter().sum();  // sequential sum for determinism
//     if y_sum == 0.0 {
//         return Err(PyValueError::new_err(
//             "[Setup BH Locs] sum(y) = 0. \nMust be non-zero for use as denominator during pdf normalization.",
//         ));
//     }
//
//     let inv_sum = 1.0 / y_sum;
//     y_slice.iter_mut().for_each(|y| *y *= inv_sum);
//
//     Ok((r_arr, r_pdf_arr))
// }


// to be used as r, r_pdf = generate_r(...)
#[pyfunction]
pub fn generate_r<'py>(
    py: Python<'py>,
    start: f64,
    end: f64,
    num_points: usize,
    crit_radius: f64,
    index_inner: f64,
    index_outer: f64,
    volume_scaling: bool,
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {

    // given that y is just an intermediary, pay attention to whether or not we actually need to
    // initialize separate vecs for it.

    let r_arr = unsafe { PyArray1::new(py, num_points, false) };
    // will be returned as_r_pdf at the end
    let r_pdf_arr = unsafe { PyArray1::new(py, num_points, false) };

    let r_slice = unsafe { r_arr.as_slice_mut().unwrap() };
    let y_slice = unsafe { r_pdf_arr.as_slice_mut().unwrap() };

    // 2. Pre-calculate constants
    // this is also the gradient from later
    let step = (end - start) / ((num_points - 1) as f64);
    let neg_inner = -index_inner;
    let neg_outer = -index_outer;
    let inv_crit = 1.0 / crit_radius;


    let split_idx = ((crit_radius - start) / step).ceil() as usize;
    let split_idx = split_idx.min(num_points);

    // removing the if/else, even though it should be pretty branch-predictor friendly
    // this allows us to do more auto-vectorization and get a decent speedup

    // what if we removed the internal if/else for volume scaling? given that the bool
    // is passed in as an argument, i would hope this is actually parametrized in some way at the
    // asm level? let's check

    // Inner region [0..split_idx]
    r_slice[..split_idx].par_iter_mut()
        .zip(y_slice[..split_idx].par_iter_mut())
        .enumerate()
        .for_each(|(i, (r_out, y_out))| {
            let val = start + (i as f64 * step);
            *r_out = val;
            let base = val * inv_crit;
            let p = base.powf(neg_inner); //  even if we dealt with the branch, this would prob
            //  still block auto-vectorization
            *y_out = if volume_scaling { PI * val * val * step * p } else { p };
        });

    // Outer region [split_idx..num_points]
    r_slice[split_idx..].par_iter_mut()
        .zip(y_slice[split_idx..].par_iter_mut())
        .enumerate()
        .for_each(|(i, (r_out, y_out))| {
            let val = start + ((i + split_idx) as f64 * step);
            *r_out = val;
            let base = val * inv_crit;
            let p = base.powf(neg_outer); // same here
            *y_out = if volume_scaling { PI * val * val * step * p } else { p };
        });


    // need to construct y first to get the sum and normalize, has to be a second pass
    let y_sum: f64 = y_slice.par_iter().sum();
    if y_sum == 0.0 {
        return Err(PyValueError::new_err("[Setup BH Locs] sum(y) = 0. \nMust be non-zero for use as denominator during pdf normalization."))
    }

    // not clear that this second pass will also benefit from being parallelized, might or might
    // not, try both
    y_slice.par_iter_mut().for_each(|y| { *y /= y_sum; });

    Ok((r_arr, r_pdf_arr))
}
