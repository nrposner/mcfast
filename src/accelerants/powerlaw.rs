use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods};
use rayon::prelude::*;

use crate::accelerants::FloatArray1;
use sleef::Sleef;           // brings pow_u10 into scope on f64
use std::f64::consts::PI;

/// Fills `r_slice` and `y_slice` over `[lo..hi)` with r-values and power-law
/// weights using exponent `neg_index`. `VOL` selects volume scaling at compile
/// time, so the branch evaporates in each monomorphization.
#[inline(always)]
fn fill_region<const VOL: bool>(
    r_slice: &mut [f64],
    y_slice: &mut [f64],
    start: f64,
    step: f64,
    inv_crit: f64,
    neg_index: f64,
    offset: usize, // global index of r_slice[0]
) {
    r_slice
        .par_iter_mut()
        .zip(y_slice.par_iter_mut())
        .enumerate()
        .for_each(|(i, (r_out, y_out))| {
            let val = start + ((i + offset) as f64) * step;
            *r_out = val;

            let base = val * inv_crit;
            // sleef scalar pow, ~1 ULP. Swap for a packed sleef call if you
            // want true SIMD; see note below.
            let p = base.powf(neg_index);

            *y_out =  PI * val * val * step * p ;
        });
}

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
    let r_arr = unsafe { PyArray1::new(py, num_points, false) };
    let r_pdf_arr = unsafe { PyArray1::new(py, num_points, false) };

    let r_slice = unsafe { r_arr.as_slice_mut().unwrap() };
    let y_slice = unsafe { r_pdf_arr.as_slice_mut().unwrap() };

    let step = (end - start) / ((num_points - 1) as f64);
    let neg_inner = -index_inner;
    let neg_outer = -index_outer;
    let inv_crit = 1.0 / crit_radius;

    let split_idx = ((crit_radius - start) / step).ceil() as usize;
    let split_idx = split_idx.min(num_points);

    // Dispatch once on the runtime bool into two fully specialized code paths.
    // Everything below this point is VOL-known-at-compile-time.
    if volume_scaling {
        let (r_in, r_out) = r_slice.split_at_mut(split_idx);
        let (y_in, y_out) = y_slice.split_at_mut(split_idx);
        fill_region::<true>(r_in, y_in, start, step, inv_crit, neg_inner, 0);
        fill_region::<true>(r_out, y_out, start, step, inv_crit, neg_outer, split_idx);
    } else {
        let (r_in, r_out) = r_slice.split_at_mut(split_idx);
        let (y_in, y_out) = y_slice.split_at_mut(split_idx);
        fill_region::<false>(r_in, y_in, start, step, inv_crit, neg_inner, 0);
        fill_region::<false>(r_out, y_out, start, step, inv_crit, neg_outer, split_idx);
    }

    let y_sum: f64 = y_slice.par_iter().sum();
    if y_sum == 0.0 {
        return Err(PyValueError::new_err(
            "[Setup BH Locs] sum(y) = 0. \nMust be non-zero for use as denominator during pdf normalization.",
        ));
    }

    let inv_sum = 1.0 / y_sum; // one divide, many muls
    y_slice.par_iter_mut().for_each(|y| *y *= inv_sum);

    Ok((r_arr, r_pdf_arr))
}


// // to be used as r, r_pdf = generate_r(...)
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
//
//     // given that y is just an intermediary, pay attention to whether or not we actually need to
//     // initialize separate vecs for it.
//
//     let r_arr = unsafe { PyArray1::new(py, num_points, false) };
//     // will be returned as_r_pdf at the end
//     let r_pdf_arr = unsafe { PyArray1::new(py, num_points, false) };
//
//     let r_slice = unsafe { r_arr.as_slice_mut().unwrap() };
//     let y_slice = unsafe { r_pdf_arr.as_slice_mut().unwrap() };
//
//     // 2. Pre-calculate constants
//     // this is also the gradient from later
//     let step = (end - start) / ((num_points - 1) as f64);
//     let neg_inner = -index_inner;
//     let neg_outer = -index_outer;
//     let inv_crit = 1.0 / crit_radius;
//
//
//     let split_idx = ((crit_radius - start) / step).ceil() as usize;
//     let split_idx = split_idx.min(num_points);
//
//     // removing the if/else, even though it should be pretty branch-predictor friendly
//     // this allows us to do more auto-vectorization and get a decent speedup
//
//     // what if we removed the internal if/else for volume scaling? given that the bool
//     // is passed in as an argument, i would hope this is actually parametrized in some way at the
//     // asm level? let's check
//
//     // Inner region [0..split_idx]
//     r_slice[..split_idx].par_iter_mut()
//         .zip(y_slice[..split_idx].par_iter_mut())
//         .enumerate()
//         .for_each(|(i, (r_out, y_out))| {
//             let val = start + (i as f64 * step);
//             *r_out = val;
//             let base = val * inv_crit;
//             let p = base.powf(neg_inner); //  even if we dealt with the branch, this would prob
//             //  still block auto-vectorization
//             *y_out = if volume_scaling { PI * val * val * step * p } else { p };
//         });
//
//     // Outer region [split_idx..num_points]
//     r_slice[split_idx..].par_iter_mut()
//         .zip(y_slice[split_idx..].par_iter_mut())
//         .enumerate()
//         .for_each(|(i, (r_out, y_out))| {
//             let val = start + ((i + split_idx) as f64 * step);
//             *r_out = val;
//             let base = val * inv_crit;
//             let p = base.powf(neg_outer); // same here
//             *y_out = if volume_scaling { PI * val * val * step * p } else { p };
//         });
//
//
//     // need to construct y first to get the sum and normalize, has to be a second pass
//     let y_sum: f64 = y_slice.par_iter().sum();
//     if y_sum == 0.0 {
//         return Err(PyValueError::new_err("[Setup BH Locs] sum(y) = 0. \nMust be non-zero for use as denominator during pdf normalization."))
//     }
//
//     // not clear that this second pass will also benefit from being parallelized, might or might
//     // not, try both
//     y_slice.par_iter_mut().for_each(|y| { *y /= y_sum; });
//
//     Ok((r_arr, r_pdf_arr))
// }
//
