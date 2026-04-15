use std::f64::consts::PI;
// use rayon::prelude::*;
use crate::accelerants::{FloatArray1, G_SI, M_SUN_KG, units::si_from_r_g};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

// good chance we can rewrite this entire thing as a single rayon iterator
// will likely benefit a lot from parallelization
// actually, will it? let's check in benchmarking

// def retro_bh_orb_disk_evolve(smbh_mass, disk_bh_retro_masses, disk_bh_retro_orbs_a, disk_bh_retro_orbs_ecc,
//                              disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse,
//                              disk_inner_stable_circ_orb, disk_surf_density_func, timestep_duration_yr, disk_radius_outer, r_g_in_meters):
//     """Evolve the orbit of initially-embedded retrograde black hole orbiters due to disk interactions.
//
//     This is a CRUDE version of evolution, future upgrades may couple to SpaceHub.
//
//     Parameters
//     ----------
//     smbh_mass : float
//         Mass [M_sun] of supermassive black hole
//     disk_bh_retro_masses : numpy.ndarray | float
//         Mass [M_sun] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     disk_bh_retro_orbs_a : numpy.ndarray
//         Orbital semi-major axes [r_{g,SMBH}] of retrograde singleton BH at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
//     disk_bh_retro_orbs_ecc : numpy.ndarray
//         Orbital eccentricity [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     disk_bh_retro_orbs_inc : numpy.ndarray
//         Orbital inclination [radians] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     disk_bh_retro_arg_periapse : numpy.ndarray
//         Argument of periapse [unitless] of retrograde singleton BH at start of a timestep with :obj:`float` type
//     disk_surf_density_func : function
//         Returns AGN gas disk surface density [kg/m^2] given a distance [r_{g,SMBH}] from the SMBH
//     timestep_duration_yr : float
//         Length of a timestep [yr]
//     r_g_in_meters: float
//         Gravitational radius of the SMBH in meters
//
//     Returns
//     -------
//     disk_bh_retro_orbs_ecc_new : numpy.ndarray
//         Updated value of eccentricity [unitless] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...
//     disk_bh_retro_orbs_a_new : numpy.ndarray
//         Updated value of semi-major axis [r_{g,SMBH}] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...
//     disk_bh_retro_orbs_inc_new : numpy.ndarray
//         Updated value of orbital inclination [radians] with :obj:`float` typeafter one timestep_duration_yr assuming gas only evolution hacked together badly...


#[pyfunction]
pub fn evolution_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64, 
    disk_bh_retro_masses_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_a_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_ecc_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_inc_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_arg_periapse_arr: PyReadonlyArray1<f64>, 
    disk_inner_stable_circ_orb: f64,
    disk_surf_arr: PyReadonlyArray1<f64>,
    disk_surf_ref_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64,
    disk_radius_outer: f64,
    rng_arr: PyReadonlyArray1<f64>
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>)> {

    // Housekeeping: if you're too close to ecc=1.0, nothing works, so
    const EPSILON: f64 = 1.0e-8;

    // hardcoded awfulness coming up:
    const SMBH_MASS_0: f64 = 1e8;  // solar masses, for scaling
    const ORBITER_MASS_0: f64 = 30.0;  // solar masses
    const PERIAPSE_1: f64 = 0.0;  // radians
    const PERIAPSE_0: f64 = PI / 2.0;  // radians

    const STEP1_ECC_0: f64 = 0.7;
    const STEP1_INC_0: f64 = PI * (175.0 / 180.0); // rad
    const STEP1_SEMI_MAJ_0: f64 = 100.0;  // r_g

    const STEP2_ECC_0: f64= 0.9999;
    const STEP2_INC_0: f64 = PI * (165.0 / 180.0);  // rad
    const STEP2_SEMI_MAJ_0: f64 = 60.0;  // r_g

    const STEP3_ECC_0: f64 = 0.9;
    const STEP3_INC_0: f64 = PI * (12.0 / 180.0);  // rad
    const STEP3_SEMI_MAJ_0: f64 = STEP2_SEMI_MAJ_0;  // r_g

    const STEP3_ECC_F: f64 = 0.5;
    const STEP3_INC_F: f64 = 0.0;  // rad
    const STEP3_SEMI_MAJ_F: f64 = 20.0;  // r_g

    const STEPW0_ECC_0: f64 = 0.7;
    const STEPW0_INC_0: f64 = PI * (175.0 / 180.0);  // rad
    const STEPW0_SEMI_MAJ_0: f64 = 100.0;  // r_g

    const STEPW0_ECC_F: f64 = 0.5;
    const STEPW0_INC_F: f64 = PI * (170.0 / 180.0); // rad
    const STEPW0_SEMI_MAJ_F: f64 = 60.0;  // r_g

    const STEP1_TIME: f64 = 1.5e5;  // years
    const STEP1_DELTA_ECC: f64 = STEP2_ECC_0 - STEP1_ECC_0;
    const STEP1_DELTA_SEMIMAJ: f64 = STEP1_SEMI_MAJ_0 - STEP2_SEMI_MAJ_0; //rg
    const STEP1_DELTA_INC: f64 = STEP1_INC_0 - STEP2_INC_0;  //rad

    const STEP2_TIME: f64 = 1.4e4;  // years
    const STEP2_DELTA_ECC: f64 = STEP2_ECC_0 - STEP3_ECC_0;
    const STEP2_DELTA_SEMIMAJ: f64 = STEP2_SEMI_MAJ_0 - STEP3_SEMI_MAJ_0;  //rg
    const STEP2_DELTA_INC: f64 = STEP2_INC_0 - STEP3_INC_0;

    const STEP3_TIME: f64 = 1.4e4;  // years
    const STEP3_DELTA_ECC: f64 = STEP3_ECC_0 - STEP3_ECC_F;
    const STEP3_DELTA_SEMIMAJ: f64 = STEP3_SEMI_MAJ_0 - STEP3_SEMI_MAJ_F;  //rg
    const STEP3_DELTA_INC: f64 = STEP3_INC_0 - STEP3_INC_F;

    const STEPW0_TIME: f64 = 1.5e7;  // years
    const STEPW0_DELTA_ECC: f64 = STEPW0_ECC_0 - STEPW0_ECC_F;
    const STEPW0_DELTA_SEMIMAJ: f64 = STEPW0_SEMI_MAJ_0 - STEPW0_SEMI_MAJ_F;  //rg
    const STEPW0_DELTA_INC: f64 = STEPW0_INC_0 - STEPW0_INC_F;

    // pre-compute SI reference constants outside the loop
    let smbh_mass_kg = smbh_mass * M_SUN_KG;
    const SMBH_MASS_0_KG: f64 = SMBH_MASS_0 * M_SUN_KG;
    const ORBITER_MASS_0_KG: f64 = ORBITER_MASS_0 * M_SUN_KG;

    // setting up slices
    let disk_bh_retro_masses_slice = disk_bh_retro_masses_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_a_slice = disk_bh_retro_orbs_a_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_ecc_slice = disk_bh_retro_orbs_ecc_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_inc_slice = disk_bh_retro_orbs_inc_arr.as_slice().unwrap();
    let disk_bh_retro_arg_periapse_slice = disk_bh_retro_arg_periapse_arr.as_slice().unwrap();
    // let disk_inner_stable_circ_orb_slice = disk_inner_stable_circ_orb_arr.as_slice().unwrap();
    let disk_surf_slice = disk_surf_arr.as_slice().unwrap();
    let disk_surf_ref_slice = disk_surf_ref_arr.as_slice().unwrap();
    let rng_slice = rng_arr.as_slice().unwrap();

    let out_a_arr = unsafe {PyArray1::new(py, disk_bh_retro_orbs_ecc_slice.len(), false)};
    let out_a_slice = unsafe {out_a_arr.as_slice_mut().unwrap()};

    let out_ecc_arr = unsafe {PyArray1::new(py, disk_bh_retro_orbs_ecc_slice.len(), false)};
    let out_ecc_slice = unsafe {out_ecc_arr.as_slice_mut().unwrap()};

    let out_inc_arr = unsafe {PyArray1::new(py, disk_bh_retro_orbs_ecc_slice.len(), false)};
    let out_inc_slice = unsafe {out_inc_arr.as_slice_mut().unwrap()};
    
    disk_bh_retro_masses_slice.iter()
        .zip(disk_bh_retro_orbs_a_slice)
        .zip(disk_bh_retro_orbs_ecc_slice)
        .zip(disk_bh_retro_orbs_inc_slice)
        .zip(disk_bh_retro_arg_periapse_slice)
        .zip(disk_surf_slice)
        .zip(disk_surf_ref_slice)
        .zip(rng_slice)
        .enumerate()
        .for_each(|(i, (((((((disk_bh_retro_masses, disk_bh_retro_orbs_a), disk_bh_retro_orbs_ecc), disk_bh_retro_orbs_inc), disk_bh_retro_arg_periapse), disk_surf), disk_surf_ref), rng))| {

        let cos_pm1_mask: bool = disk_bh_retro_arg_periapse.cos().abs() >= 0.5;
        let cos_0_mask: bool = !cos_pm1_mask;

        // these two are related but not exhaustive
        let no_max_ecc_retro_mask: bool = (disk_bh_retro_orbs_ecc < &STEP2_ECC_0) & (disk_bh_retro_orbs_inc.abs() >= PI/2.0); 
        let max_ecc_mask: bool = disk_bh_retro_orbs_ecc >= &STEP2_ECC_0;
        let barely_prograde_mask: bool = disk_bh_retro_orbs_inc.abs() < PI/2.0;
        let ecc_unreliable_mask: bool = !(no_max_ecc_retro_mask | max_ecc_mask | barely_prograde_mask);
        if ecc_unreliable_mask {
            // todo: improve error handling
            // Err(PyValueError::new_err("ECC Warning: retrograde orbital parameters out of range, behavior unreliable"))
            panic!("ECC Warning: retrograde orbital parameters out of range, behavior unreliable")
        };

        let condition1 = cos_pm1_mask & no_max_ecc_retro_mask;
        let condition2 = cos_pm1_mask & max_ecc_mask;
        let condition3 = cos_pm1_mask & barely_prograde_mask;
        let conditionw0 = cos_0_mask;

        if condition1 | condition2 | condition3 | conditionw0 {

            let periapse = if cos_pm1_mask {PERIAPSE_1} else if cos_0_mask {PERIAPSE_0} else {-100.5};

            // Condition priority matches Python's masked-assignment order (last write wins):
            // condition3 (barely_prograde) must be checked before condition2 (max_ecc)
            // because they can overlap (ecc >= 0.9999 AND |inc| < pi/2), and in
            // Python the barely_prograde assignment comes after max_ecc, overwriting it.
            let (semi_maj_0, ecc0, inc0, time, delta_ecc, delta_semimaj, delta_inc) = if condition1 {
                (STEP1_SEMI_MAJ_0, STEP1_ECC_0, STEP1_INC_0, STEP1_TIME, STEP1_DELTA_ECC, STEP1_DELTA_SEMIMAJ, STEP1_DELTA_INC)
            } else if condition3 {
                (STEP3_SEMI_MAJ_0, STEP3_ECC_0, STEP3_INC_0, STEP3_TIME, STEP3_DELTA_ECC, STEP3_DELTA_SEMIMAJ, STEP3_DELTA_INC)
            } else if condition2 {
                (STEP2_SEMI_MAJ_0, STEP2_ECC_0, STEP2_INC_0, STEP2_TIME, STEP2_DELTA_ECC, STEP2_DELTA_SEMIMAJ, STEP2_DELTA_INC)
            } else {
                (STEPW0_SEMI_MAJ_0, STEPW0_ECC_0, STEPW0_INC_0, STEPW0_TIME, STEPW0_DELTA_ECC, STEPW0_DELTA_SEMIMAJ, STEPW0_DELTA_INC)
            };

            // check that this is the right input
            let semi_maj_axis = si_from_r_g(smbh_mass, *disk_bh_retro_orbs_a);

            let retro_mass_kg = disk_bh_retro_masses * M_SUN_KG;

            let (tau_e_current, tau_a_current) = tau_ecc_dyn_local(smbh_mass_kg, retro_mass_kg, *disk_bh_retro_orbs_ecc, *disk_bh_retro_orbs_inc, *disk_bh_retro_arg_periapse, *disk_surf, semi_maj_axis);
            let tau_inc_current = tau_inc_dyn_local(smbh_mass_kg, retro_mass_kg, *disk_bh_retro_orbs_ecc, *disk_bh_retro_orbs_inc, *disk_bh_retro_arg_periapse, *disk_surf, semi_maj_axis);

            let semi_maj_0_si = si_from_r_g(SMBH_MASS_0, semi_maj_0);

            let (tau_e_ref, tau_a_ref) = tau_ecc_dyn_local(SMBH_MASS_0_KG, ORBITER_MASS_0_KG, ecc0, inc0, periapse, *disk_surf_ref, semi_maj_0_si);
            let tau_inc_ref = tau_inc_dyn_local(SMBH_MASS_0_KG, ORBITER_MASS_0_KG, ecc0, inc0, periapse, *disk_surf_ref, semi_maj_0_si);

            let tau_e_div = tau_e_current / tau_e_ref;
            let tau_a_div = tau_a_current / tau_a_ref;
            let tau_inc_div = tau_inc_current / tau_inc_ref;

            let (semimaj_scale_factor, inc_scale_factor) = (time * tau_a_div, time * tau_inc_div);

            // Guard for ecc=0 (circular orbit): tau_e_dyn is mathematically undefined
            // when ecc=0 because tau_a = tau_p, giving 0 * 1/|1/tau_a - 1/tau_p| = 0 * inf = NaN.
            // A circular orbit has no eccentricity to damp, so ecc stays at its current value.
            let disk_bh_retro_orbs_ecc_new: f64 = if *disk_bh_retro_orbs_ecc < EPSILON {
                *disk_bh_retro_orbs_ecc
            } else {
                let ecc_scale_factor = time * tau_e_div;
                (disk_bh_retro_orbs_ecc * (
                    1.0 - delta_ecc / disk_bh_retro_orbs_ecc * (timestep_duration_yr / ecc_scale_factor))
                ).clamp(0.0, 1.0-EPSILON)
            };
            let disk_bh_retro_orbs_a_new: f64 = (
                disk_bh_retro_orbs_a * (
                    1.0 - delta_semimaj / disk_bh_retro_orbs_a * (timestep_duration_yr / semimaj_scale_factor)
                )).clamp(disk_inner_stable_circ_orb, f64::INFINITY);
            let disk_bh_retro_orbs_inc_new: f64 = (
                disk_bh_retro_orbs_inc * (
                    1.0 - delta_inc / disk_bh_retro_orbs_inc * (timestep_duration_yr / inc_scale_factor)
                )).clamp(0.0, f64::INFINITY);

            // properly speaking, this could be part of the clamp operation, but the behavior is
            // slightly different
            // disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new > *disk_radius_outer] = disk_radius_outer - epsilon_orb_a[disk_bh_retro_orbs_a_new > *disk_radius_outer]
            let disk_bh_retro_orbs_a_new = if disk_bh_retro_orbs_a_new > disk_radius_outer {
                let epsilon_orb_a = disk_radius_outer * 
                    ((disk_bh_retro_masses / (3.0 * (disk_bh_retro_masses + smbh_mass))).cbrt()) * rng;
                disk_radius_outer - epsilon_orb_a
            } else {
                disk_bh_retro_orbs_a_new
            };

            // Finite check: if any output is non-finite, apply fallback
            if !disk_bh_retro_orbs_ecc_new.is_finite()
                || !disk_bh_retro_orbs_a_new.is_finite()
                || !disk_bh_retro_orbs_inc_new.is_finite()
            {
                if *disk_bh_retro_orbs_a < 12.1 {
                    // Inside ACTUAL ISCO; preserve old ecc, mark as eaten
                    out_ecc_slice[i] = *disk_bh_retro_orbs_ecc;
                    out_a_slice[i] = 5.9;
                    out_inc_slice[i] = 0.0;
                } else {
                    out_ecc_slice[i] = 2.0;
                    out_a_slice[i] = 0.0;
                    out_inc_slice[i] = 0.0;
                    panic!(
                        "Finite check failed: i={}, ecc={}, mass={}, a={}, inc={}, periapse={}",
                        i, disk_bh_retro_orbs_ecc, disk_bh_retro_masses,
                        disk_bh_retro_orbs_a, disk_bh_retro_orbs_inc, disk_bh_retro_arg_periapse
                    );
                }
            } else {
                out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
                out_a_slice[i] = disk_bh_retro_orbs_a_new;
                out_inc_slice[i] = disk_bh_retro_orbs_inc_new;
            }
        } else {
            out_ecc_slice[i] = 0.0;
            out_a_slice[i] = 0.0;
            out_inc_slice[i] = 0.0;
        }
    });
    Ok((out_ecc_arr, out_a_arr, out_inc_arr))
}

pub fn tau_ecc_dyn_local(
    smbh_mass: f64,
    // in kg
    // contrary to documentation, this is either an ndarray, 
    // or a float, exactly half the time
    retro_mass: f64,
    ecc: f64,
    inc: f64,
    omega: f64,
    disk_surf: f64,
    semi_maj_axis: f64
) -> (f64, f64) {
    let cos_omega = omega.cos();

    let period = 2.0 * PI * (semi_maj_axis.powi(3) / (G_SI * smbh_mass)).sqrt();
    let rec = semi_maj_axis * (1.0 - (ecc.powi(2)));
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

    let tau_p_dyn_denom_1 = retro_mass * disk_surf * PI * rec.powi(2);

    let tau_p_dyn_denom_2 = 2.0f64.sqrt() * kappa * (inc.cos() - zeta).abs();

    let tau_p_dyn_total = tau_p_dyn_numerator_part / tau_p_dyn_denom_1 / tau_p_dyn_denom_2;

    let tau_a_dyn = tau_p_dyn_total * (1.0 - ecc.powi(2)) * kappa * (inc.cos() - zeta).abs() / (kappa_bar * (inc.cos() - zeta_bar).abs());
    let tau_e_dyn = (2.0 * ecc.powi(2) / (1.0 - ecc.powi(2))) * 1.0 / (1.0 / tau_a_dyn - 1.0 / tau_p_dyn_total).abs();

    (tau_e_dyn, tau_a_dyn)
}


pub fn tau_inc_dyn_local(
    smbh_mass: f64,
    orbiter_mass: f64,
    ecc: f64,
    inc: f64,
    omega: f64,
    disk_surf: f64,
    semi_maj_axis: f64,
) -> f64 {
    let static_period_component = G_SI * smbh_mass;
    let cos_omega = omega.cos();

    let period = 2.0 * PI * (semi_maj_axis.powi(3) / static_period_component).sqrt();
    let rec = semi_maj_axis * (1.0 - (ecc.powi(2)));
    let sigma_plus = (1.0 + ecc.powi(2) + 2.0 * ecc * cos_omega).sqrt();
    let sigma_minus = (1.0 + ecc.powi(2) - 2.0 * ecc * cos_omega).sqrt();
    let eta_plus = (1.0 + ecc * cos_omega).sqrt();
    let eta_minus = (1.0 - ecc * cos_omega).sqrt();
    let kappa = 0.5 * ( (1.0 / (eta_plus.powi(15))).sqrt() + ((1.0 / eta_minus).powi(15)).sqrt());
    let delta = 0.5 * ( sigma_plus / (eta_plus.powi(2)) + sigma_minus / (eta_minus.powi(2)) );

    let tau_i_dyn_1 = 2.0f64.sqrt() * inc * ((delta - inc.cos()).powf(1.5));
    let tau_i_dyn_2 = smbh_mass.powi(2) * period;

    let tau_i_dyn_denom_chunk = orbiter_mass * disk_surf * PI * rec.powi(2);

    // Python: (Num / Denom_Chunk) / kappa
    (tau_i_dyn_1 * tau_i_dyn_2) / tau_i_dyn_denom_chunk / kappa
}
