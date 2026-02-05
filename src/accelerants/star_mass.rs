use std::f64::consts::PI;

use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use crate::accelerants::luminosity::si_from_r_g;

// const M_SUN_KG: f64 = 1.9884099e30;  // Solar mass in kg
// const C_SI: f64 = 299792460.0;     // Speed of light in m/s
// const G_SI: f64 = 6.67430e-11;     // Gravitational constant in m^3/(kg s^2)
use crate::accelerants::{C_SI, FloatArray1, G_SI, L_SUN_W, M_SUN_KG, R_SUN_M, YR_S};

#[pyfunction]
pub fn star_wind_mass_loss_helper<'py>(
    py: Python<'py>,
    disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_star_pro_log_radius_arr: PyReadonlyArray1<f64>,
    disk_star_pro_log_lum_arr: PyReadonlyArray1<f64>,
    disk_opacity_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64
) -> (Bound<'py, PyArray1<f64>>, f64) {
    let disk_star_pro_masses_slice = disk_star_pro_masses_arr.as_slice().unwrap();
    let disk_star_pro_log_radius_slice = disk_star_pro_log_radius_arr.as_slice().unwrap();
    let disk_star_pro_log_lum_slice = disk_star_pro_log_lum_arr.as_slice().unwrap();
    let disk_opacity_slice = disk_opacity_arr.as_slice().unwrap();

    let star_new_masses_arr = unsafe { PyArray1::new(py, disk_star_pro_masses_slice.len(), false) };
    let star_new_masses_slice = unsafe { star_new_masses_arr.as_slice_mut().unwrap() };

    let timestep_s = timestep_duration_yr * YR_S;

    let mut mass_lost_acc = 0.0_f64; // accumulates in Msun

    for (i, (((star_mass_msun, log_radius), log_lum), disk_opacity)) in disk_star_pro_masses_slice.iter()
        .zip(disk_star_pro_log_radius_slice)
        .zip(disk_star_pro_log_lum_slice)
        .zip(disk_opacity_slice)
        .enumerate()
    {
        // Convert everything to SI up front
        let star_mass_kg = star_mass_msun * M_SUN_KG;
        let star_radius_m = 10.0f64.powf(*log_radius) * R_SUN_M;
        let star_lum_w = 10.0f64.powf(*log_lum) * L_SUN_W;
        // disk_opacity is already in m²/kg

        // Eddington luminosity (watts)
        let l_edd_w = 4.0 * PI * G_SI * C_SI * star_mass_kg / disk_opacity;

        // Escape speed (m/s)
        let v_esc = (2.0 * G_SI * star_mass_kg / star_radius_m).sqrt();

        // Dimensionless tanh argument (both numerator and denominator in watts)
        let tanh_argument = (star_lum_w - l_edd_w) / (0.1 * l_edd_w);

        // Mass loss rate (kg/s), already negative
        let mdot = -(star_lum_w / v_esc.powi(2)) * (1.0 + f64::tanh(tanh_argument));

        // Mass lost this timestep, converted to Msun
        let mass_lost_msun = (mdot * timestep_s) / M_SUN_KG;

        mass_lost_acc += mass_lost_msun;

        let star_new_mass = star_mass_msun + mass_lost_msun;
        debug_assert!(star_new_mass > 0.0, "star_new_mass <= 0 at index {i}");

        star_new_masses_slice[i] = star_new_mass;
    }

    (star_new_masses_arr, mass_lost_acc)
}


#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn accrete_star_mass_helper<'py>(
    py: Python<'py>,
    disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_star_pro_orbs_arr: PyReadonlyArray1<f64>,
    // ???
    disk_star_luminosity_factor: f64,
    disk_star_initial_mass_cutoff: f64,
    smbh_mass: f64,
    sound_speed_arr: PyReadonlyArray1<f64>,
    disk_density_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64,
    // r_g_in_meters: f64,
) -> (FloatArray1<'py>, f64, f64) {
    let disk_star_pro_masses_slice = disk_star_pro_masses_arr.as_slice().unwrap();
    let disk_star_pro_orbs_slice = disk_star_pro_orbs_arr.as_slice().unwrap();
    let sound_speed_slice = sound_speed_arr.as_slice().unwrap();
    let disk_density_slice = disk_density_arr.as_slice().unwrap();

    let star_new_masses_arr = unsafe { PyArray1::new(py, disk_star_pro_masses_slice.len(), false) };
    let star_new_masses_slice = unsafe { star_new_masses_arr.as_slice_mut().unwrap() };

    let mut mass_gained_acc = 0.0f64;
    let mut immortal_mass_lost_acc = 0.0f64;

    for (i, (((star_mass, orb), sound_speed), density)) in disk_star_pro_orbs_slice.iter()
        .zip(disk_star_pro_orbs_slice)
        .zip(sound_speed_slice)
        .zip(disk_density_slice)
        .enumerate() {

        // turn to meters (already there, no?)
        let r_bondi = 2.0 * G_SI * star_mass / sound_speed.powi(2);
        
        // oddly, two different versions of star_mass are being used, with one being in solar
        // masses and the other potentially not??
        let r_hill_rg = orb * (star_mass / (3.0 * (star_mass + smbh_mass))).powf(1.0/3.0);
        let r_hll_m = si_from_r_g(smbh_mass, r_hill_rg);

        let min_radius = r_bondi.min(r_hll_m);

        // in kg / year
        let mdot = (PI / disk_star_luminosity_factor) * density * sound_speed * min_radius.powi(2);

        // accrete mass onto stars
        // to Msun
        let new_mass = (star_mass + mdot * timestep_duration_yr).clamp(0.0, disk_star_initial_mass_cutoff);

        // to Msun
        let mass_gained = mdot * timestep_duration_yr;

        let immortal_mass_lost = if new_mass == disk_star_initial_mass_cutoff {
            let immortal_mass_diff = new_mass - star_mass;
            mass_gained - immortal_mass_diff
        } else {
            0.0
        };

        mass_gained_acc += mass_gained;
        immortal_mass_lost_acc += immortal_mass_lost;

        star_new_masses_slice[i] = new_mass;
    }

    (star_new_masses_arr, mass_gained_acc, immortal_mass_lost_acc)
}
