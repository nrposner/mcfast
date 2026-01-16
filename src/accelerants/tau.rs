use std::f64::consts::PI;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::constants::G;

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn tau_ecc_dyn_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    // in kg
    // contrary to documentation, this is either an ndarray, 
    // or a float, exactly half the time
    retro_mass: &Bound<'_, PyAny>,
    ecc_arr: PyReadonlyArray1<f64>,
    inc_arr: PyReadonlyArray1<f64>,
    omega_arr: PyReadonlyArray1<f64>,
    disk_surf_res_arr: PyReadonlyArray1<f64>,
    semi_maj_axis_arr: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {

    let semi_maj_axis_slice = semi_maj_axis_arr.as_slice().unwrap();
    let ecc_slice = ecc_arr.as_slice().unwrap();
    let inc_slice = inc_arr.as_slice().unwrap();
    let omega_slice = omega_arr.as_slice().unwrap();
    let disk_surf_res_slice = disk_surf_res_arr.as_slice().unwrap();

    let tau_e_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
    let tau_e_dyn_slice = unsafe {tau_e_dyn_arr.as_slice_mut().unwrap()};

    let tau_a_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
    let tau_a_dyn_slice = unsafe {tau_a_dyn_arr.as_slice_mut().unwrap()};

    // can be either an array of length N, or a scalar, for silly reasons
    if let Ok(retro_mass_arr) = retro_mass.extract::<PyReadonlyArray1<f64>>() {

        let retro_mass_slice = retro_mass_arr.as_slice().unwrap();

        for (i, (((((semi, ecc), omega), inc), rm), ds)) in semi_maj_axis_slice.iter()
            .zip(ecc_slice)
            .zip(omega_slice)
            .zip(inc_slice)
            .zip(retro_mass_slice)
            .zip(disk_surf_res_slice)
            .enumerate() { 

            let cos_omega = omega.cos();

            let period = 2.0 * PI * (semi.powi(3) / (G * smbh_mass)).sqrt();
            let rec = semi * (1.0 - (ecc.powi(2)));
            let sigma_plus = (1.0 + ecc.powi(2) + 2.0 * ecc * cos_omega).sqrt();
            let sigma_minus = (1.0 + ecc.powi(2) - 2.0 * ecc * cos_omega).sqrt();
            let eta_plus = (1.0 + ecc * cos_omega).sqrt();
            let eta_minus = (1.0 - ecc * cos_omega).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let xi = 0.5 * ( (1.0 / (eta_plus.powi(13))).sqrt() + (1.0 / (eta_minus.powi(13))).sqrt() );
            let zeta = xi/kappa;
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

            let kappa_bar = 0.5 * ( (1.0 / (eta_plus.powi(7))).sqrt() + ((1.0 / eta_minus).powi(7)).sqrt());
            let xi_bar = 0.5 * (
                ( sigma_plus.powi(4) / eta_plus.powi(13) ).sqrt() + ( sigma_minus.powi(4) / eta_minus.powi(13) ).sqrt()
            );
            let zeta_bar = xi_bar/kappa_bar;

            let tau_p_dyn_numerator_part = inc.sin().abs() * ((delta - inc.cos()).powf(1.5)) * smbh_mass.powi(2) * period;

            let tau_p_dyn_denom_1 = rm * ds * PI * rec.powi(2);

            let tau_p_dyn_factor_2 = 2.0f64.sqrt();
            let tau_p_dyn_multipliers = kappa * (inc.cos() - zeta).abs();

            let tau_p_dyn_total = (tau_p_dyn_numerator_part / tau_p_dyn_denom_1) / tau_p_dyn_factor_2 * tau_p_dyn_multipliers;

            let tau_a_dyn = tau_p_dyn_total * (1.0 - ecc.powi(2)) * kappa * (inc.cos() - zeta).abs() / (kappa_bar * (inc.cos() - zeta_bar).abs());
            let tau_e_dyn = (2.0 * ecc.powi(2) / (1.0 - ecc.powi(2))) * 1.0 / (1.0 / tau_a_dyn - 1.0 / tau_p_dyn_total).abs();

            tau_e_dyn_slice[i] = tau_e_dyn;
            tau_a_dyn_slice[i] = tau_a_dyn;
        }
        Ok((tau_e_dyn_arr, tau_a_dyn_arr))

    } else if let Ok(retro_mass_scalar) = retro_mass.extract::<f64>() {

        for (i, ((((semi, ecc), omega), inc), ds)) in semi_maj_axis_slice.iter()
            .zip(ecc_slice)
            .zip(omega_slice)
            .zip(inc_slice)
            .zip(disk_surf_res_slice)
            .enumerate() { 

            let cos_omega = omega.cos();

            let period = 2.0 * PI * (semi.powi(3) / (G * smbh_mass)).sqrt();
            let rec = semi * (1.0 - (ecc.powi(2)));
            let sigma_plus = (1.0 + ecc.powi(2) + 2.0 * ecc * cos_omega).sqrt();
            let sigma_minus = (1.0 + ecc.powi(2) - 2.0 * ecc * cos_omega).sqrt();
            let eta_plus = (1.0 + ecc * cos_omega).sqrt();
            let eta_minus = (1.0 - ecc * cos_omega).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let xi = 0.5 * ( (1.0 / (eta_plus.powi(13))).sqrt() + (1.0 / (eta_minus.powi(13))).sqrt() );
            let zeta = xi/kappa;
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

            let kappa_bar = 0.5 * ( (1.0 / (eta_plus.powi(7))).sqrt() + ((1.0 / eta_minus).powi(7)).sqrt());
            let xi_bar = 0.5 * (
                ( sigma_plus.powi(4) / eta_plus.powi(13) ).sqrt() + ( sigma_minus.powi(4) / eta_minus.powi(13) ).sqrt()
            );
            let zeta_bar = xi_bar/kappa_bar;

            let tau_p_dyn_numerator_part = inc.sin().abs() * ((delta - inc.cos()).powf(1.5)) * smbh_mass.powi(2) * period;

            let tau_p_dyn_denom_1 = retro_mass_scalar * ds * PI * rec.powi(2);

            let tau_p_dyn_factor_2 = 2.0f64.sqrt();
            let tau_p_dyn_multipliers = kappa * (inc.cos() - zeta).abs();

            // Replicating Python: (Num / Denom1) / Sqrt(2) * Multipliers
            let tau_p_dyn_total = (tau_p_dyn_numerator_part / tau_p_dyn_denom_1) / tau_p_dyn_factor_2 * tau_p_dyn_multipliers;

            let tau_a_dyn = tau_p_dyn_total * (1.0 - ecc.powi(2)) * kappa * (inc.cos() - zeta).abs() / (kappa_bar * (inc.cos() - zeta_bar).abs());
            let tau_e_dyn = (2.0 * ecc.powi(2) / (1.0 - ecc.powi(2))) * 1.0 / (1.0 / tau_a_dyn - 1.0 / tau_p_dyn_total).abs();

            tau_e_dyn_slice[i] = tau_e_dyn;
            tau_a_dyn_slice[i] = tau_a_dyn;
        }
        
        Ok((tau_e_dyn_arr, tau_a_dyn_arr))

    } else {
        Err(PyValueError::new_err("Input `retro_mass derived from retrograde_bh_masses is neither a numeric scalar nor a numpy ndarray."))
        
    }
}


#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn tau_inc_dyn_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    orbiter_mass_obj: &Bound<'_, PyAny>,
    ecc_arr: PyReadonlyArray1<f64>,
    inc_arr: PyReadonlyArray1<f64>,
    cos_omega: PyReadonlyArray1<f64>,
    disk_surf_res_arr: PyReadonlyArray1<f64>,
    semi_maj_axis_arr: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {

    let semi_maj_axis_slice = semi_maj_axis_arr.as_slice().unwrap();
    let ecc_slice = ecc_arr.as_slice().unwrap();
    let inc_slice = inc_arr.as_slice().unwrap();
    let cos_omega_slice = cos_omega.as_slice().unwrap();
    let disk_surf_res_slice = disk_surf_res_arr.as_slice().unwrap();

    let out_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
    let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

    // TODO:
    // this could be made even more efficient by pulling out more of the
    // static parts of the period calculation
    let static_period_component = G * smbh_mass;

    // part of: let period = 2.0 * PI * (semi.powi(3) / (G * smbh_mass)).sqrt();

    if let Ok(orbiter_mass_arr) = orbiter_mass_obj.extract::<PyReadonlyArray1<f64>>() {

        let orbiter_mass_slice = orbiter_mass_arr.as_slice().unwrap();

        for (i, (((((semi, e), co), inc), om), ds)) in semi_maj_axis_slice.iter()
            .zip(ecc_slice)
            .zip(cos_omega_slice)
            .zip(inc_slice)
            .zip(orbiter_mass_slice)
            .zip(disk_surf_res_slice)
            .enumerate() { 

            let period = 2.0 * PI * (semi.powi(3) / static_period_component).sqrt();
            let rec = semi * (1.0 - e.powi(2));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );
            
            let tau_i_dyn_1 = 2.0f64.sqrt() * inc * ((delta - inc.cos()).powf(1.5));
            let tau_i_dyn_2 = smbh_mass.powi(2) * period;

            let tau_i_dyn_denom_chunk = om * ds * PI * rec.powi(2);

            // Python: (Num / Denom_Chunk) / kappa
            out_slice[i] = (tau_i_dyn_1 * tau_i_dyn_2) / tau_i_dyn_denom_chunk / kappa;
        }
        Ok(out_arr)

    } else if let Ok(orbiter_mass_scalar) = orbiter_mass_obj.extract::<f64>() {

        for (i, ((((semi, e), co), inc), ds)) in semi_maj_axis_slice.iter()
            .zip(ecc_slice)
            .zip(cos_omega_slice)
            .zip(inc_slice)
            .zip(disk_surf_res_slice)
            .enumerate() { 

            let period = 2.0 * PI * (semi.powi(3) / static_period_component).sqrt();
            let rec = semi * (1.0 - (e.powi(2)));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

            let tau_i_dyn_1 = 2.0f64.sqrt() * inc * ((delta - inc.cos()).powf(1.5));
            let tau_i_dyn_2 = smbh_mass.powi(2) * period;

            let tau_i_dyn_denom_chunk = orbiter_mass_scalar * ds * PI * rec.powi(2);

            // Python: (Num / Denom_Chunk) / kappa
            out_slice[i] = (tau_i_dyn_1 * tau_i_dyn_2) / tau_i_dyn_denom_chunk / kappa;
        }
        Ok(out_arr)

    } else {
        Err(PyValueError::new_err("Input `retro_mass derived from retrograde_bh_masses is neither a numeric scalar nor a numpy ndarray."))
    }
}

/// A function to output fizzbuzz when provided with the 
/// maximum number i to which to count
fn write_fizzbuzz(i: u32) {

}
