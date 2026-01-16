use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use rayon::prelude::*;

#[pyfunction]
pub fn continuous_broken_powerlaw<'py>(
    py: Python<'py>,
    radius: PyReadonlyArray1<'py, f64>,
    crit_radius: f64,
    // should be an int, but it seems that a float is being passed in instead
    index: f64,
) -> Bound<'py, PyArray1<f64>> {
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
) -> Bound<'py, PyArray1<f64>> {
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

#[pyfunction]
pub fn dual_powerlaw_with_grid<'py>(
    py: Python<'py>,
    start: f64,
    end: f64,
    num_points: usize,
    crit_radius: f64,
    index_inner: f64,
    index_outer: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
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



// #[pyfunction]
// pub fn dual_powerlaw<'py>(
//     py: Python<'py>,
//     radius: PyReadonlyArray1<'py, f64>,
//     crit_radius: f64,
//     index_inner: f64,
//     index_outer: f64,
// ) -> Bound<'py, PyArray1<f64>> {
//     let radius_slice = radius.as_slice().unwrap();
//
//     // Create uninitialized array (faster than generic new which might zero-fill)
//     let out_arr = unsafe { PyArray1::new(py, radius_slice.len(), false) };
//     let out_slice = unsafe { out_arr.as_slice_mut().unwrap() };
//
//     // Pre-calculate negative indices to save 1 negation per loop
//     let neg_inner = -index_inner;
//     let neg_outer = -index_outer;
//     // Pre-calculate inverse crit_radius to use multiplication (faster than division)
//     let inv_crit = 1.0 / crit_radius;
//
//     for (i, r) in radius_slice.iter().enumerate() {
//         if *r <= crit_radius {
//             out_slice[i] = (r * inv_crit).powf(neg_inner);
//         } else {
//             out_slice[i] = (r * inv_crit).powf(neg_outer);
//         }
//     }
//     out_arr
// }

