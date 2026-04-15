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

    // 3. Parallel fused generation
    // We zip the two output slices together to write to them simultaneously
    r_slice.par_iter_mut()
        .zip(y_slice.par_iter_mut())
        .enumerate()
        .for_each(|(i, (r_out, y_out))| {
            // Calculate grid point
            let val = start + (i as f64 * step);
            *r_out = val;

            // Calculate powerlaw immediately (value is hot in register)
            // the inner if/else check is always the same for each function call,
            // so the branch predictor should be fine, but double check
            if val <= crit_radius {
                if volume_scaling {
                    let shell_volume = PI * val.powi(2) * step;
                    *y_out = (val * inv_crit).powf(neg_inner) * shell_volume;
                } else {
                    *y_out = (val * inv_crit).powf(neg_inner);
                }
            } else if volume_scaling {
                let shell_volume = PI * val.powi(2) * step;
                *y_out = (val * inv_crit).powf(neg_outer) * shell_volume;
            } else {
                *y_out = (val * inv_crit).powf(neg_outer);
            }
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

