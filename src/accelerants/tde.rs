use crate::accelerants::units::si_from_r_g;

use pyo3::{prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};


// def check_tde_or_flip(star_retro_id_num, star_retro_mass, star_retro_log_radius, star_retro_orb_ecc, star_retro_orb_a, smbh_mass, r_g_in_meters):
//     """Retrograde stars that flip to prograde are TDEs if they are inside the disk's tidal disruption radius and have sufficiently high eccentricity.
//
//     Parameters
//     ----------
//     star_retro_id_num : numpy.ndarray
//         ID numbers of retrograde stars that may flip to prograde
//     star_retro_mass : numpy.ndarray
//         Star mass [Msun] with :obj:`float` type
//     star_retro_log_radius : numpy.ndarray
//         Star log radius/Rsun with :obj:`float` type
//     star_retro_orb_ecc : numpy.ndarray
//         Star orbital eccentricity
//     star_retro_orb_a : numpy.ndarray
//         Star semi-major axis wrt SMBH with :obj:`float` type
//     smbh_mass : float
//         Mass [Msun] of the SMBH
//     r_g_in_meters: float
//         Gravitational radius of the SMBH in meters
//
//     Returns
//     -------
//     tde_id_num : numpy.ndarray
//         ID numbers of stars that will become TDEs
//     flip_id_num : numpy.ndarray
//         ID numbers of stars that will flip to prograde
//     """
//
//     # Convert everything to units
//     star_mass = star_retro_mass * u.Msun
//     star_radius = (10 ** star_retro_log_radius) * u.Rsun
//     # star_orb_a = (si_from_r_g(smbh_mass, star_retro_orb_a, r_g_defined=r_g_in_meters)).to("meter")
//     star_orb_a = si_from_r_g_optimized(smbh_mass, star_retro_orb_a)
//     smbh_mass_units = smbh_mass * u.Msun
//
//     # Tidal disruption radius of the disk is R_star * (M_smbh / M_star)^1/3
//     disk_radius_tidal_disruption = (star_radius * ((smbh_mass_units / star_mass) ** (1./3.))).to("meter")
//
//     # Stars are TDEs if they are inside the TD radius and have eccentricity >= 0.8
//     tde_mask = ((star_orb_a * (1. - star_retro_orb_ecc)) <= disk_radius_tidal_disruption) & (star_retro_orb_ecc >= 0.8)
//
//     # Stars that don't become TDEs will flip to prograde
//     tde_id_num = star_retro_id_num[tde_mask]
//     flip_id_num = star_retro_id_num[~tde_mask]
//
//     return (tde_id_num, flip_id_num)

#[pyfunction]
pub fn tde_helper<'py>(
    py: Python<'py>, 
    star_retro_id_num_arr: PyReadonlyArray1<u64>, 
    star_retro_mass_arr: PyReadonlyArray1<f64>, 
    star_retro_log_radius_arr: PyReadonlyArray1<f64>, 
    star_retro_orb_ecc_arr: PyReadonlyArray1<f64>, 
    star_retro_orb_a_arr: PyReadonlyArray1<f64>, 
    smbh_mass: f64
) -> (Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>) {

    let star_retro_id_num_slice = star_retro_id_num_arr.as_slice().unwrap();
    let star_retro_mass_slice = star_retro_mass_arr.as_slice().unwrap();
    let star_retro_log_radius_slice = star_retro_log_radius_arr.as_slice().unwrap();
    let star_retro_orb_ecc_slice = star_retro_orb_ecc_arr.as_slice().unwrap();
    let star_retro_orb_a_slice = star_retro_orb_a_arr.as_slice().unwrap();

    let n = star_retro_id_num_slice.len();

    // Allocate numpy arrays at the max possible size, uninitialized.
    let tde_arr = unsafe { PyArray1::<u64>::new(py, [n], false) };
    let flip_arr = unsafe { PyArray1::<u64>::new(py, [n], false) };

    let tde_ptr = unsafe { tde_arr.as_slice_mut().unwrap() };
    let flip_ptr = unsafe { flip_arr.as_slice_mut().unwrap() };

    let mut tde_len = 0usize;
    let mut flip_len = 0usize;

    for ((((id_num, mass), log_radius), orb_ecc), orb_a) in star_retro_id_num_slice.iter()
        .zip(star_retro_mass_slice)
        .zip(star_retro_log_radius_slice)
        .zip(star_retro_orb_ecc_slice)
        .zip(star_retro_orb_a_slice)
    {
        let star_radius = 10.0f64.powf(*log_radius);
        let star_orb_a = si_from_r_g(smbh_mass, *orb_a);
        let disk_radius_tidal_disruption = star_radius * (smbh_mass / mass).cbrt();
        let tde_mask = (star_orb_a * (1.0 - *orb_ecc) <= disk_radius_tidal_disruption) && (*orb_ecc >= 0.8);
        
        if tde_mask {
            tde_ptr[tde_len] = *id_num;
            tde_len += 1;
        } else {
            flip_ptr[flip_len] = *id_num;
            flip_len += 1;
        }
    }

    // Shrink numpy arrays to actual used size.
    // Use numpy's resize — requires refcheck=false since we hold the only ref.
    unsafe {
        tde_arr.resize([tde_len]).unwrap();
        flip_arr.resize([flip_len]).unwrap();
    }

    (tde_arr, flip_arr)
}


#[pyfunction]
pub fn tde_helper_variant<'py>(
    py: Python<'py>, 
    star_retro_id_num_arr: PyReadonlyArray1<u64>, 
    star_retro_mass_arr: PyReadonlyArray1<f64>, 
    star_retro_log_radius_arr: PyReadonlyArray1<f64>, 
    star_retro_orb_ecc_arr: PyReadonlyArray1<f64>, 
    star_retro_orb_a_arr: PyReadonlyArray1<f64>, 
    smbh_mass: f64
) -> (Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>) {

    let star_retro_id_num_slice = star_retro_id_num_arr.as_slice().unwrap();
    let star_retro_mass_slice = star_retro_mass_arr.as_slice().unwrap();
    let star_retro_log_radius_slice = star_retro_log_radius_arr.as_slice().unwrap();
    let star_retro_orb_ecc_slice = star_retro_orb_ecc_arr.as_slice().unwrap();
    let star_retro_orb_a_slice = star_retro_orb_a_arr.as_slice().unwrap();

    let n = star_retro_id_num_slice.len();

    // First pass: count how many go to each bucket.
    let mut tde_count = 0usize;
    for i in 0..n {
        let star_radius = 10.0f64.powf(star_retro_log_radius_slice[i]);
        let star_orb_a = si_from_r_g(smbh_mass, star_retro_orb_a_slice[i]);
        let disk_radius_td = star_radius * (smbh_mass / star_retro_mass_slice[i]).cbrt();
        let ecc = star_retro_orb_ecc_slice[i];
        if (star_orb_a * (1.0 - ecc) <= disk_radius_td) && (ecc >= 0.8) {
            tde_count += 1;
        }
    }
    let flip_count = n - tde_count;

    // Allocate numpy arrays at exact final size.
    let tde_arr = PyArray1::<u64>::zeros(py, [tde_count], false);
    let flip_arr = PyArray1::<u64>::zeros(py, [flip_count], false);
    let tde_slice = unsafe { tde_arr.as_slice_mut().unwrap() };
    let flip_slice = unsafe { flip_arr.as_slice_mut().unwrap() };

    // Second pass: fill. Recomputing the predicate is usually cheaper than
    // allocating a mask vec, unless the predicate is expensive.
    let mut ti = 0;
    let mut fi = 0;
    for i in 0..n {
        let star_radius = 10.0f64.powf(star_retro_log_radius_slice[i]);
        let star_orb_a = si_from_r_g(smbh_mass, star_retro_orb_a_slice[i]);
        let disk_radius_td = star_radius * (smbh_mass / star_retro_mass_slice[i]).cbrt();
        let ecc = star_retro_orb_ecc_slice[i];
        let id = star_retro_id_num_slice[i];
        if (star_orb_a * (1.0 - ecc) <= disk_radius_td) && (ecc >= 0.8) {
            tde_slice[ti] = id; ti += 1;
        } else {
            flip_slice[fi] = id; fi += 1;
        }
    }

    (tde_arr, flip_arr)
}
