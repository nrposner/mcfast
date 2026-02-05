#![allow(dead_code)]

// setup mstar runs
// dont worry abt setup scripts, gives us a bunch of mass bins?
// some of these, especially at the higher mass end, the higher the mas sof ht egalazy, the longer
// it takes to run, some of these take 3 hours to run
// just bc there's more objects in the disk
//
// barry would also be interested in seeing how dynamics, where the bottlenecks are there
// which scripts? Just mcfacts_sim
//
// make plots is the most common according to harry


use pyo3::prelude::*;
use std::f64::consts::PI;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::accelerants::{C_CGS, C_SI, FloatArray1, G_CGS, G_SI, M_SUN_KG};

/// Calculate the gravitational radius r_g in SI units (meters)
/// This matches the Python si_from_r_g function behavior
pub fn si_from_r_g(smbh_mass: f64, distance_rg: f64) -> f64 {
    // smbh_mass is in solar masses, convert to kg
    let smbh_mass_kg = smbh_mass * M_SUN_KG;
    
    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);
    
    // Calculate distance in meters
    distance_rg * r_g
}

#[pyfunction]
pub fn shock_luminosity_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    mass_final_arr: PyReadonlyArray1<f64>,
    bin_orb_arr: PyReadonlyArray1<f64>,
    disk_height_arr: PyReadonlyArray1<f64>,
    disk_density_arr: PyReadonlyArray1<f64>,
    v_kick_arr: PyReadonlyArray1<f64>,
) -> FloatArray1<'py> {

    let mass_final_slice = mass_final_arr.as_slice().unwrap();
    let bin_orb_slice = bin_orb_arr.as_slice().unwrap();
    let disk_height_slice = disk_height_arr.as_slice().unwrap();
    let disk_density_slice = disk_density_arr.as_slice().unwrap();
    let v_kick_slice = v_kick_arr.as_slice().unwrap();

    let out_arr = unsafe { PyArray1::new(py, mass_final_slice.len(), false) };
    let out_slice = unsafe { out_arr.as_slice_mut().unwrap() };

    // Initialize scaling values from McKernan et al. (2019)
    let r_hill_rg_scale = 1000.0_f64 * ((65.0 / 1e9f64) / 3.0).powf(1.0/3.0);
    let r_hill_mass_scale = M_SUN_KG;
    let v_kick_scale = 100.0;

    for (i, ((((mass_final, bin_orb), disk_height_rg), disk_density_si), v_kick)) in 
        mass_final_slice.iter()
            .zip(bin_orb_slice)
            .zip(disk_height_slice)
            .zip(disk_density_slice)
            .zip(v_kick_slice)
            .enumerate() 
    { 
        // Get the Hill radius in [R_g] and convert to [m]
        let r_hill_rg = bin_orb * ((mass_final / smbh_mass) / 3.0).powf(1.0/3.0); 
        let r_hill_m = si_from_r_g(smbh_mass, r_hill_rg);

        // Get the height of the disk in [m]
        // disk_height_rg is already computed as disk_aspect_ratio(bin_orb_a) * bin_orb_a
        let disk_height_m = si_from_r_g(smbh_mass, *disk_height_rg);

        // Compute the volume of the Hill sphere in [m^3]
        let v_hill = (4.0 / 3.0) * PI * r_hill_m.powi(3);
        
        // Compute the volume of the gas contained within the hill sphere [m^3]
        let r_minus_h = r_hill_m - disk_height_m;
        let v_subtrahend = (2.0 / 3.0) * PI * r_minus_h.powi(2) * (3.0 * r_hill_m - r_minus_h);
        let v_hill_gas = (v_hill - v_subtrahend).abs();

        // Use the disk density and volume to get the mass of gas in the Hill sphere [kg]
        // disk_density_si is already in [kg/m^3]
        let r_hill_mass = disk_density_si * v_hill_gas;

        // Calculate the energy dissipated into the disk [erg]
        let energy = 1e47 * (r_hill_mass / r_hill_mass_scale) * (v_kick / v_kick_scale).powi(2);

        // Calculate the time scale for energy dissipation [s]
        let time = 1.577e7 * (r_hill_rg / 3.0 * r_hill_rg_scale) / (v_kick / v_kick_scale);

        // Calculate the shock luminosity [erg/s]
        out_slice[i] = energy / time;
    }

    out_arr
}


#[pyfunction]
pub fn jet_luminosity_helper<'py>(
    py: Python<'py>,
    mass_final_arr: PyReadonlyArray1<f64>,
    disk_density_arr: PyReadonlyArray1<f64>,
    spin_final_arr: PyReadonlyArray1<f64>,
    v_kick_arr: PyReadonlyArray1<f64>,
    sound_speed_arr: PyReadonlyArray1<f64>,
) -> FloatArray1<'py> {
    let mass_final_slice = mass_final_arr.as_slice().unwrap();
    let disk_density_slice = disk_density_arr.as_slice().unwrap();
    let spin_final_slice = spin_final_arr.as_slice().unwrap();
    let v_kick_slice = v_kick_arr.as_slice().unwrap();
    let sound_speed_slice = sound_speed_arr.as_slice().unwrap();

    let out_arr = unsafe { PyArray1::new(py, mass_final_slice.len(), false) };
    let out_slice = unsafe { out_arr.as_slice_mut().unwrap() };
    
    for (i, ((((mass_final, disk_density), spin_final), v_kick), sound_speed)) in 
        mass_final_slice.iter()
            .zip(disk_density_slice)
            .zip(spin_final_slice)
            .zip(v_kick_slice)
            .zip(sound_speed_slice)
            .enumerate() {

        let v_rel = v_kick * 1e5;
        let mass_final_g = mass_final * 1.98841e33;

        let mdot_bondi = 4.0 * PI * G_CGS.powi(2) * mass_final_g.powi(2) * disk_density * (v_rel.powi(2) + (sound_speed * 100.0).powi(2)).powf(-3.0/2.0);

        let kappa = 0.1;

        let l_jet = (0.1) * (kappa / 0.1) * (0.9f64 / spin_final).powi(2) * mdot_bondi * C_CGS.powi(2);

        out_slice[i] = l_jet;
    }

    out_arr
}
