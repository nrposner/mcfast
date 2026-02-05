use std::f64::consts::PI;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use rayon::prelude::*;

use crate::accelerants::FloatArray1;

#[pyfunction]
pub fn continuous_broken_powerlaw<'py>(
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    crit_radius: f64,
    // should be an int, but it seems that a float is being passed in instead
    index: f64,
) -> FloatArray1<'py> {
    let radius_slice = radius.as_slice().unwrap();

    let out_arr = unsafe{ PyArray1::new(py, radius_slice.len(), false)};
    let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

    for (i, r) in radius_slice.iter().enumerate() {
        out_slice[i] = (r/crit_radius).powf(-index) 
    }
    out_arr
}


// intended to be used as: `y_scaled = dual_powerlaw(radius, crit_radius, inner, outer)`
#[pyfunction]
pub fn dual_powerlaw<'py>(
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    crit_radius: f64,
    index_inner: f64,
    index_outer: f64,
) -> FloatArray1<'py> {
    let radius_slice = radius.as_slice().unwrap();
    
    let out_arr = unsafe { PyArray1::new(py, radius_slice.len(), false) };
    let out_slice = unsafe { out_arr.as_slice_mut().unwrap() };

    // 1. Find the partition point. 
    // Since r is sorted, we find the first index where r > crit_radius
    // let split_idx = radius_slice.partition_point(|&r| r <= crit_radius);

    let neg_inner = -index_inner;
    let neg_outer = -index_outer;
    let inv_crit = 1.0 / crit_radius;

    out_slice.par_iter_mut()
        .zip(radius_slice.par_iter())
        .for_each(|(out, &r)| {
            if r <= crit_radius {
                *out = (r * inv_crit).powf(neg_inner);
            } else {
                *out = (r * inv_crit).powf(neg_outer);
            }
        });


    // // 2. Inner Loop (0 to split) - No branches inside
    // // LLVM loves this and will likely auto-vectorize it
    // for (i, r) in radius_slice[..split_idx].iter().enumerate() {
    //     out_slice[i] = (r * inv_crit).powf(neg_inner);
    // }
    //
    // // 3. Outer Loop (split to end) - No branches inside
    // for (i, r) in radius_slice[split_idx..].iter().enumerate() {
    //     out_slice[split_idx + i] = (r * inv_crit).powf(neg_outer);
    // }

    out_arr
}

// intended to be used as
// r, y_unscaled = dual_powerlaw_with_grid(disk_inner_stable_circ_orb, disk_radius_outer, 1000000, nsc_radius_crit_rg, nsc_density_index_inner, nsc_density_index_outer)
#[pyfunction]
pub fn dual_powerlaw_with_grid<'py>(
    py: Python<'py>,
    start: f64,
    end: f64,
    num_points: usize,
    crit_radius: f64,
    index_inner: f64,
    index_outer: f64,
) -> (FloatArray1<'py>, FloatArray1<'py>) {
    // 1. Allocate uninitialized output arrays
    let r_arr = unsafe { PyArray1::new(py, num_points, false) };
    let y_arr = unsafe { PyArray1::new(py, num_points, false) };
    
    let r_slice = unsafe { r_arr.as_slice_mut().unwrap() };
    let y_slice = unsafe { y_arr.as_slice_mut().unwrap() };

    // 2. Pre-calculate constants
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
            if val <= crit_radius {
                *y_out = (val * inv_crit).powf(neg_inner);
            } else {
                *y_out = (val * inv_crit).powf(neg_outer);
            }
        });

    (r_arr, y_arr)
}

// to be used as r, r_pdf = generate_r(...)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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

