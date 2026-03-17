use std::f64::consts::PI;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods};
use rayon::prelude::*;

use crate::accelerants::FloatArray1;

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

    // Inner region [0..split_idx]
    r_slice[..split_idx].par_iter_mut()
        .zip(y_slice[..split_idx].par_iter_mut())
        .enumerate()
        .for_each(|(i, (r_out, y_out))| {
            let val = start + (i as f64 * step);
            *r_out = val;
            let base = val * inv_crit;
            let p = base.powf(neg_inner);
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
            let p = base.powf(neg_outer);
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

