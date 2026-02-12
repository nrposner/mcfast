use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use std::f64::consts::PI;
use crate::accelerants::{C_SI, FloatArray1, G_SI, M_SUN_KG, MPC_SI};

#[pyfunction(signature=(mass_1_obj, mass_2_arr, obj_sep_arr, timestep_duration_yr, old_gw_freq_arr, smbh_mass, agn_redshift, flag_include_old_gw_freq))]
pub fn gw_strain_helper<'py>(
    py: Python<'py>,
    mass_1_obj: &Bound<'_, PyAny>, // PyReadonlyArray1<f64>,  
    mass_2_arr: PyReadonlyArray1<f64>,
    obj_sep_arr: PyReadonlyArray1<f64>, 
    timestep_duration_yr: f64, 
    old_gw_freq_arr: PyReadonlyArray1<f64>, 
    smbh_mass: f64, 
    agn_redshift: f64,
    flag_include_old_gw_freq: bool, // defaults to true?
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {


    if let Ok(mass_1_arr) = mass_1_obj.extract::<PyReadonlyArray1<f64>>() {

        let mass_1_slice = mass_1_arr.as_slice().unwrap();
        let mass_2_slice = mass_2_arr.as_slice().unwrap();
        let obj_sep_slice = obj_sep_arr.as_slice().unwrap();
        let old_gw_freq_slice = old_gw_freq_arr.as_slice().unwrap();

        let char_strain_arr = unsafe { PyArray1::new(py, mass_1_slice.len(), false) };
        let char_strain_slice = unsafe { char_strain_arr.as_slice_mut().unwrap() };

        let nu_gw_arr = unsafe { PyArray1::new(py, mass_1_slice.len(), false) };
        let nu_gw_slice = unsafe { nu_gw_arr.as_slice_mut().unwrap() };

        // turn years into seconds
        // not quite 365*24*60*60, slightly higher
        let timestep_units = timestep_duration_yr * 31557600.0;

        // rg is in meters
        let rg = 1.5e11 * (smbh_mass/1e8f64);

        for (i, (((mass_1, mass_2), obj_sep), old_gw_freq)) in mass_1_slice.iter()
            .zip(mass_2_slice)
            .zip(obj_sep_slice)
            .zip(old_gw_freq_slice)
            .enumerate() {

            // cds Msun is just 1.98840987e+30 kg, same as M_SUN_KG
            let mass_1 = mass_1 * M_SUN_KG;
            let mass_2 = mass_2 * M_SUN_KG;

            let mass_total = mass_1 + mass_2;

            let bin_sep = obj_sep * rg;

            let mass_chirp = ((mass_1 * mass_2).powf(3.0/5.0)) / (mass_total.powf(1.0/5.0));

            // already in meters
            let rg_chirp = (G_SI * mass_chirp) / C_SI.powi(2);

            let bin_sep = bin_sep.max(rg_chirp);

            // already in Hz
            let nu_gw = (1.0 / PI) * (mass_total * G_SI / bin_sep.powi(3)).sqrt();

            let d_obs = match agn_redshift {
                0.1 => 421.0 * MPC_SI,
                0.5 => 1909.0 * MPC_SI,
                _ => panic!("The only valid values for agn_redshift are {{0.1, 0.5}}, not {}", agn_redshift),
            };

            let strain = (4.0/d_obs) * rg_chirp * (PI * nu_gw * rg_chirp / C_SI).powf(2.0/3.0);

            // gwb isn't used in the original code, can take it out entirely
            // let gwb = nu_gw > 2.0e-3f64;

            let tight = nu_gw > 2e-3;
            let lessonesix = nu_gw < 1e-6;
            let greateronesix = nu_gw > 1e-6;

            // should precisely match the current logic of the strain_factor creation, though
            // the fallthrough for flag false and nu_gw == 1e-6 is inelegant and may be revised
            let strain_factor = match (flag_include_old_gw_freq, tight, lessonesix, greateronesix) {
                (true, false, _, _) => {
                    let delta_nu = (old_gw_freq - nu_gw).abs();
                    let delta_nu_delta_timestep = delta_nu / timestep_units;
                    let nu_squared = nu_gw.powi(2);
                    // nu_factor is also not used in the original code
                    // let nu_factor = nu_gw.powf(-5.0/6.0);

                    ((nu_squared / delta_nu_delta_timestep) / 8.0).sqrt()
                },
                (true, true, _, _) => {
                    let num_factor = (5.0f64/96.0f64).sqrt() * (1.0/(8.0*PI)) * (1.0 / PI).powf(1.0/3.0);
                    num_factor * ((C_SI / rg_chirp).powf(5.0/6.0)) * (nu_gw).powf(-5.0/6.0)
                },
                (false, _, false, true) => 4.0e3,
                (false, _, true, false) => {
                    (nu_gw * PI * 1e7 / 8.0).sqrt()
                }
                _ => 1.0
            };

            let char_strain = strain_factor*strain;

            char_strain_slice[i] = char_strain;
            nu_gw_slice[i] = nu_gw;
        }
        Ok((char_strain_arr, nu_gw_arr))

    } else if let Ok(mass_1) = mass_1_obj.extract::<f64>() {
        let mass_2_slice = mass_2_arr.as_slice().unwrap();
        let obj_sep_slice = obj_sep_arr.as_slice().unwrap();
        let old_gw_freq_slice = old_gw_freq_arr.as_slice().unwrap();

        let char_strain_arr = unsafe { PyArray1::new(py, mass_2_slice.len(), false) };
        let char_strain_slice = unsafe { char_strain_arr.as_slice_mut().unwrap() };

        let nu_gw_arr = unsafe { PyArray1::new(py, mass_2_slice.len(), false) };
        let nu_gw_slice = unsafe { nu_gw_arr.as_slice_mut().unwrap() };

        // turn years into seconds
        // not quite 365*24*60*60, slightly higher
        let timestep_units = timestep_duration_yr * 31557600.0;

        // rg is in meters
        let rg = 1.5e11 * (smbh_mass/1e8f64);

        for (i, ((mass_2, obj_sep), old_gw_freq)) in mass_2_slice.iter()
            .zip(obj_sep_slice)
            .zip(old_gw_freq_slice)
            .enumerate() {

            // cds Msun is just 1.98840987e+30 kg, same as M_SUN_KG
            let mass_1 = mass_1 * M_SUN_KG;
            let mass_2 = mass_2 * M_SUN_KG;

            let mass_total = mass_1 + mass_2;

            let bin_sep = obj_sep * rg;

            let mass_chirp = ((mass_1 * mass_2).powf(3.0/5.0)) / (mass_total.powf(1.0/5.0));

            // already in meters
            let rg_chirp = (G_SI * mass_chirp) / C_SI.powi(2);

            let bin_sep = bin_sep.max(rg_chirp);

            // already in Hz
            let nu_gw = (1.0 / PI) * (mass_total * G_SI / bin_sep.powi(3)).sqrt();

            let d_obs = match agn_redshift {
                0.1 => 421.0 * MPC_SI,
                0.5 => 1909.0 * MPC_SI,
                _ => panic!("The only valid values for agn_redshift are {{0.1, 0.5}}, not {}", agn_redshift),
            };

            let strain = (4.0/d_obs) * rg_chirp * (PI * nu_gw * rg_chirp / C_SI).powf(2.0/3.0);

            // gwb isn't used in the original code, can take it out entirely
            // let gwb = nu_gw > 2.0e-3f64;

            let tight = nu_gw > 2e-3;
            let lessonesix = nu_gw < 1e-6;
            let greateronesix = nu_gw > 1e-6;

            // should precisely match the current logic of the strain_factor creation, though
            // the fallthrough for flag false and nu_gw == 1e-6 is inelegant and may be revised
            let strain_factor = match (flag_include_old_gw_freq, tight, lessonesix, greateronesix) {
                (true, false, _, _) => {
                    let delta_nu = (old_gw_freq - nu_gw).abs();
                    let delta_nu_delta_timestep = delta_nu / timestep_units;
                    let nu_squared = nu_gw.powi(2);
                    // nu_factor is also not used in the original code
                    // let nu_factor = nu_gw.powf(-5.0/6.0);

                    ((nu_squared / delta_nu_delta_timestep) / 8.0).sqrt()
                },
                (true, true, _, _) => {
                    let num_factor = (5.0f64/96.0f64).sqrt() * (1.0/(8.0*PI)) * (1.0 / PI).powf(1.0/3.0);
                    num_factor * ((C_SI / rg_chirp).powf(5.0/6.0)) * (nu_gw).powf(-5.0/6.0)
                },
                (false, _, false, true) => 4.0e3,
                (false, _, true, false) => {
                    (nu_gw * PI * 1e7 / 8.0).sqrt()
                }
                _ => 1.0
            };

            let char_strain = strain_factor*strain;

            char_strain_slice[i] = char_strain;
            nu_gw_slice[i] = nu_gw;
        }
        Ok((char_strain_arr, nu_gw_arr))

    } else {
        Err(PyValueError::new_err("Input `retro_mass derived from retrograde_bh_masses is neither a numeric scalar nor a numpy ndarray."))
        
    }
}
