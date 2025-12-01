use rayon::iter::IndexedParallelIterator;
use std::f64::consts::PI;

use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use rayon::iter::IntoParallelRefIterator;
// use uom;

use crate::cubes::G;

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn tau_p_dyn_rs<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    // in kg
    // contrary to documentation, this is either an ndarray, 
    // or a float, exactly half the time
    // retro_mass: PyReadonlyArray1<f64>,
    retro_mass: &Bound<'_, PyAny>,
    ecc: PyReadonlyArray1<f64>,
    inc: PyReadonlyArray1<f64>,
    omega: PyReadonlyArray1<f64>,
    disk_surf_res: PyReadonlyArray1<f64>,
    semi_maj_axis: PyReadonlyArray1<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {

    if let Ok(retro_mass_arr) = retro_mass.extract::<PyReadonlyArray1<f64>>() {

        let retro_mass_slice = retro_mass_arr.as_slice().unwrap();

        let semi_maj_axis_slice = semi_maj_axis.as_slice().unwrap();
        let ecc_slice = ecc.as_slice().unwrap();
        let inc_slice = inc.as_slice().unwrap();
        // let cos_omega_slice = cos_omega.as_slice().unwrap();
        let omega_slice = omega.as_slice().unwrap();
        let disk_surf_res_slice = disk_surf_res.as_slice().unwrap();

        let tau_e_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let tau_e_dyn_slice = unsafe {tau_e_dyn_arr.as_slice_mut().unwrap()};

        let tau_a_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let tau_a_dyn_slice = unsafe {tau_a_dyn_arr.as_slice_mut().unwrap()};
        // let out_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        // let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

        for (i, (((((semi, e), o), inc), rm), ds)) in semi_maj_axis_slice.iter().zip(ecc_slice).zip(omega_slice).zip(inc_slice).zip(retro_mass_slice).zip(disk_surf_res_slice).enumerate() { 
            // cos_omega
            let co = o.cos();
            let period = (semi.powi(3) / G * smbh_mass).sqrt();
            let rec = semi * (1.0 - (e.powi(2)));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let xi = 0.5 * ( (1.0 / (eta_plus.powi(13))).sqrt() + (1.0 / (eta_minus.powi(13))).sqrt() );
            let zeta = xi/kappa;
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

            let kappa_bar = 0.5 * ( (1.0 / (eta_plus.powi(7))).sqrt() + ((1.0 / eta_minus).powi(7)).sqrt());
            let xi_bar = 0.5 * (
                ( sigma_plus.powi(4) / eta_plus.powi(13) ).sqrt() + ( sigma_minus.powi(4) / eta_minus.powi(13) ).sqrt()
            );
            let zeta_bar = xi_bar/kappa_bar;

            // going line by line
            let tau_p_dyn_1 = inc.sin().abs() * ((delta - inc.cos()).powf(1.5));
            let tau_p_dyn_2 = smbh_mass.powi(2) * period;
            let tau_p_dyn_3 = rm * ds * PI * rec.powi(2);
            let tau_p_dyn_4 = 2.0f64.sqrt() * kappa * (inc.cos() - zeta).abs();

            let tau_p_dyn_total = tau_p_dyn_1 * tau_p_dyn_2 / tau_p_dyn_3 / tau_p_dyn_4;

            let tau_a_dyn = tau_p_dyn_total * (1.0 - e.powi(2)) * kappa * (inc.cos() - zeta).abs() / (kappa_bar * (inc.cos() - zeta_bar).abs());
            let tau_e_dyn = (2.0 * e.powi(2) / (1.0 - e.powi(2))) * 1.0 / (1.0 / tau_a_dyn - 1.0 / tau_p_dyn_total).abs();

            tau_e_dyn_slice[i] = tau_e_dyn;
            tau_a_dyn_slice[i] = tau_a_dyn;

            // out_slice[i] = tau_p_dyn_1 * tau_p_dyn_2 / tau_p_dyn_3 / tau_p_dyn_4;
        }
        // Ok(out_arr)
        Ok((tau_e_dyn_arr, tau_a_dyn_arr))

    } else if let Ok(retro_mass_scalar) = retro_mass.extract::<f64>() {

        let semi_maj_axis_slice = semi_maj_axis.as_slice().unwrap();
        let ecc_slice = ecc.as_slice().unwrap();
        let inc_slice = inc.as_slice().unwrap();
        // let cos_omega_slice = cos_omega.as_slice().unwrap();
        let omega_slice = omega.as_slice().unwrap();
        let disk_surf_res_slice = disk_surf_res.as_slice().unwrap();

        let tau_e_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let tau_e_dyn_slice = unsafe {tau_e_dyn_arr.as_slice_mut().unwrap()};

        let tau_a_dyn_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let tau_a_dyn_slice = unsafe {tau_a_dyn_arr.as_slice_mut().unwrap()};

        for (i, ((((semi, e), o), inc), ds)) in semi_maj_axis_slice.iter().zip(ecc_slice).zip(omega_slice).zip(inc_slice).zip(disk_surf_res_slice).enumerate() { 
            let co = o.cos();
            let period = (semi.powi(3) / G * smbh_mass).sqrt();
            let rec = semi * (1.0 - (e.powi(2)));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let xi = 0.5 * ( (1.0 / (eta_plus.powi(13))).sqrt() + (1.0 / (eta_minus.powi(13))).sqrt() );
            let zeta = xi/kappa;
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

            let kappa_bar = 0.5 * ( (1.0 / (eta_plus.powi(7))).sqrt() + ((1.0 / eta_minus).powi(7)).sqrt());
            let xi_bar = 0.5 * (
                ( sigma_plus.powi(4) / eta_plus.powi(13) ).sqrt() + ( sigma_minus.powi(4) / eta_minus.powi(13) ).sqrt()
            );
            let zeta_bar = xi_bar/kappa_bar;
            
            // going line by line
            let tau_p_dyn_1 = inc.sin().abs() * ((delta - inc.cos()).powf(1.5));
            let tau_p_dyn_2 = smbh_mass.powi(2) * period;
            let tau_p_dyn_3 = retro_mass_scalar * ds * PI * rec.powi(2);
            let tau_p_dyn_4 = 2.0f64.sqrt() * kappa * (inc.cos() - zeta).abs();

            let tau_p_dyn_total = tau_p_dyn_1 * tau_p_dyn_2 / tau_p_dyn_3 / tau_p_dyn_4;

            let tau_a_dyn = tau_p_dyn_total * (1.0 - e.powi(2)) * kappa * (inc.cos() - zeta).abs() / (kappa_bar * (inc.cos() - zeta_bar).abs());
            let tau_e_dyn = (2.0 * e.powi(2) / (1.0 - e.powi(2))) * 1.0 / (1.0 / tau_a_dyn - 1.0 / tau_p_dyn_total).abs();

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
pub fn tau_inc_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    // in kg
    // contrary to documentation, this is either an ndarray, 
    // or a float, exactly half the time
    // retro_mass: PyReadonlyArray1<f64>,
    // retro_mass: &Bound<'_, PyAny>,
    orbiter_mass: &Bound<'_, PyAny>,
    ecc: PyReadonlyArray1<f64>,
    inc: PyReadonlyArray1<f64>,
    cos_omega: PyReadonlyArray1<f64>,
    disk_surf_res: PyReadonlyArray1<f64>,
    semi_maj_axis: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {

    if let Ok(orbiter_mass_arr) = orbiter_mass.extract::<PyReadonlyArray1<f64>>() {

        let orbiter_mass_slice = orbiter_mass_arr.as_slice().unwrap();

        let semi_maj_axis_slice = semi_maj_axis.as_slice().unwrap();
        let ecc_slice = ecc.as_slice().unwrap();
        let inc_slice = inc.as_slice().unwrap();
        let cos_omega_slice = cos_omega.as_slice().unwrap();
        let disk_surf_res_slice = disk_surf_res.as_slice().unwrap();

        let out_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

        for (i, (((((semi, e), co), inc), om), ds)) in semi_maj_axis_slice.iter().zip(ecc_slice).zip(cos_omega_slice).zip(inc_slice).zip(orbiter_mass_slice).zip(disk_surf_res_slice).enumerate() { 
            let period = (semi.powi(3) / G * smbh_mass).sqrt();
            let rec = semi * (1.0 - (e.powi(2)));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );
            
            // going line by line
            
        // tau_i_dyn = np.sqrt(2.0) * inc * ((delta - np.cos(inc)) ** 1.5) \
        //             * (SI_smbh_mass ** 2) * period / (
        //                         SI_orbiter_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) / kappa

            let tau_i_dyn_1 = 2.0f64.sqrt() * inc * ((delta - inc.cos()).powf(1.5));
            let tau_i_dyn_2 = smbh_mass.powi(2) * period;
            let tau_i_dyn_3 = om * ds * PI * rec.powi(2) / kappa;

            out_slice[i] = tau_i_dyn_1 * tau_i_dyn_2 / tau_i_dyn_3
        }
        
        Ok(out_arr)
    } else if let Ok(orbiter_mass_scalar) = orbiter_mass.extract::<f64>() {
        let semi_maj_axis_slice = semi_maj_axis.as_slice().unwrap();
        let ecc_slice = ecc.as_slice().unwrap();
        let inc_slice = inc.as_slice().unwrap();
        let cos_omega_slice = cos_omega.as_slice().unwrap();
        let disk_surf_res_slice = disk_surf_res.as_slice().unwrap();

        let out_arr = unsafe{ PyArray1::new(py, semi_maj_axis_slice.len(), false)};
        let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

        for (i, ((((semi, e), co), inc), ds)) in semi_maj_axis_slice.iter().zip(ecc_slice).zip(cos_omega_slice).zip(inc_slice).zip(disk_surf_res_slice).enumerate() { 
            let period = (semi.powi(3) / G * smbh_mass).sqrt();
            let rec = semi * (1.0 - (e.powi(2)));
            let sigma_plus = (1.0 + e.powi(2) + 2.0 * e * co).sqrt();
            let sigma_minus = (1.0 + e.powi(2) - 2.0 * e * co).sqrt();
            let eta_plus = (1.0 + e * co).sqrt();
            let eta_minus = (1.0 - e * co).sqrt();
            let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
            let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );
            
            // going line by line
            
        // tau_i_dyn = np.sqrt(2.0) * inc * ((delta - np.cos(inc)) ** 1.5) \
        //             * (SI_smbh_mass ** 2) * period / (
        //                         SI_orbiter_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) / kappa

            let tau_i_dyn_1 = 2.0f64.sqrt() * inc * ((delta - inc.cos()).powf(1.5));
            let tau_i_dyn_2 = smbh_mass.powi(2) * period;
            let tau_i_dyn_3 = orbiter_mass_scalar * ds * PI * rec.powi(2) / kappa;

            out_slice[i] = tau_i_dyn_1 * tau_i_dyn_2 / tau_i_dyn_3
        }
        
        Ok(out_arr)

    } else {
        Err(PyValueError::new_err("Input `retro_mass derived from retrograde_bh_masses is neither a numeric scalar nor a numpy ndarray."))
    }
}

    // # throw most things into SI units (that's right, ENGINEER UNITS!)
    // #    or more locally convenient variable names
    // SI_smbh_mass = smbh_mass * u.Msun.to("kg")  # kg
    // SI_semi_maj_axis = si_from_r_g(smbh_mass, disk_bh_retro_orbs_a, r_g_defined=r_g_in_meters).to("m").value
    // SI_orbiter_mass = disk_bh_retro_masses * u.Msun.to("kg")  # kg
    // omega = disk_bh_retro_arg_periapse  # radians
    // ecc = disk_bh_retro_orbs_ecc  # unitless
    // inc = disk_bh_retro_orbs_inc  # radians
    // cos_omega = np.cos(omega)
    //
    // # period in units of sec
    // period = 2.0 * np.pi * np.sqrt((SI_semi_maj_axis ** 3) / (const.G * SI_smbh_mass))
    // # semi-latus rectum in units of meters
    // semi_lat_rec = SI_semi_maj_axis * (1.0 - (ecc ** 2))
    // # WZL Eqn 7 (sigma+/-)
    // sigma_plus = np.sqrt(1.0 + (ecc ** 2) + 2.0 * ecc * cos_omega)
    // sigma_minus = np.sqrt(1.0 + (ecc ** 2) - 2.0 * ecc * cos_omega)
    // # WZL Eqn 8 (eta+/-)
    // eta_plus = np.sqrt(1.0 + ecc * cos_omega)
    // eta_minus = np.sqrt(1.0 - ecc * cos_omega)
    // # WZL Eqn 62
    // kappa = 0.5 * (np.sqrt(1.0 / (eta_plus ** 15)) + np.sqrt(1.0 / (eta_minus ** 15)))
    // # WZL Eqn 30
    // delta = 0.5 * (sigma_plus / (eta_plus ** 2) + sigma_minus / (eta_minus ** 2))
    // # WZL Eqn 71
    // #   NOTE: preserved disk_bh_retro_orbs_a in r_g to feed to disk_surf_density_func function
    // #   tau in units of sec
    // tau_i_dyn = np.sqrt(2.0) * inc * ((delta - np.cos(inc)) ** 1.5) \
    //             * (SI_smbh_mass ** 2) * period / (
    //                         SI_orbiter_mass * disk_surf_density_func(disk_bh_retro_orbs_a) * np.pi * (semi_lat_rec ** 2)) \
    //             / kappa


// smbh mass in this call chain is taken from opts.smbh_mass and not converted into any additional
// units until this point

// def tau_semi_lat(smbh_mass, retrograde_bh_locations, retrograde_bh_masses, retrograde_bh_orb_ecc, retrograde_bh_orb_inc,
//                  retro_arg_periapse, disk_surf_model, r_g_in_meters):
//     """Calculates how fast the semi-latus rectum of a retrograde single orbiter changes due to dynamical friction
//
//     Parameters
//     ----------
//     smbh_mass : float
//         Mass [M_sun] of supermassive black hole
//     retrograde_bh_locations : numpy.ndarray
//         Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
//     retrograde_bh_masses : numpy.ndarray
//         Mass [M_sun] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     retrograde_bh_orb_ecc : numpy.ndarray
//         Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     retrograde_bh_orb_inc : numpy.ndarray
//         Orbital inclination [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     retro_arg_periapse : numpy.ndarray
//         Argument of periapse [radian] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     disk_surf_model : function
//         Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
//     r_g_in_meters: float
//         Gravitational radius of the SMBH in meters
//
//     Returns
//     -------
//     tau_p_dyn : numpy.ndarray
//         Timescale [s] for the evolution of the semi-latus rectum of each object

