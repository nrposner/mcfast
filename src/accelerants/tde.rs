use std::f64::consts::PI;
use crate::accelerants::units::si_from_r_g;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::accelerants::{FloatArray1, G_SI};


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
pub fn tde_helper_helper<'py>(
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

    let (tde_id_num, flip_id_num): (Vec<u64>, Vec<u64>) = star_retro_id_num_slice.iter()
        .zip(star_retro_mass_slice)
        .zip(star_retro_log_radius_slice)
        .zip(star_retro_orb_ecc_slice)
        .zip(star_retro_orb_a_slice)
        .fold((Vec::new(), Vec::new()), |(mut tde_id_num, mut flip_id_num), ((((id_num, mass), log_radius), orb_ecc), orb_a)| {
            let star_radius = 10.0f64.powf(*log_radius);

            let star_orb_a = si_from_r_g(smbh_mass, *orb_a);

            // need to check units here
            let disk_radius_tidal_disruption = star_radius * (smbh_mass / mass).cbrt(); // evaluate
            // to meter??

            let tde_mask = ((star_orb_a * (1.0 - *orb_ecc)) <= disk_radius_tidal_disruption) & (*orb_ecc >= 0.8);
            if tde_mask { tde_id_num.push(*id_num); } else { flip_id_num.push(*id_num); }

            (tde_id_num, flip_id_num)
    });
    
    let tde_arr = PyArray1::from_vec(py, tde_id_num);
    let flip_arr = PyArray1::from_vec(py, flip_id_num);

    (tde_arr, flip_arr)
}

