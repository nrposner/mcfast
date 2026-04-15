use std::f64::consts::PI;
use rayon::prelude::*;
use crate::accelerants::{G_SI, units::si_from_r_g};
use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

// good chance we can rewrite this entire thing as a single rayon iterator
// will likely benefit a lot from parallelization


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
pub fn retro_bh_orb_disk_evolve_opt<'py>(
    py: Python<'py>,
    smbh_mass: f64, 
    disk_bh_retro_masses_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_a_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_ecc_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_orbs_inc_arr: PyReadonlyArray1<f64>, 
    disk_bh_retro_arg_periapse_arr: PyReadonlyArray1<f64>, 
    disk_inner_stable_circ_orb_arr: PyReadonlyArray1<f64>, 
    disk_surf_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64, 
    disk_radius_outer_arr: PyReadonlyArray1<f64>,
    rng_arr: PyReadonlyArray1<f64>
) {
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

    // setting up slices
    let disk_bh_retro_masses_slice = disk_bh_retro_masses_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_a_slice = disk_bh_retro_orbs_a_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_ecc_slice = disk_bh_retro_orbs_ecc_arr.as_slice().unwrap();
    let disk_bh_retro_orbs_inc_slice = disk_bh_retro_orbs_inc_arr.as_slice().unwrap();
    let disk_bh_retro_arg_periapse_slice = disk_bh_retro_arg_periapse_arr.as_slice().unwrap();
    let disk_inner_stable_circ_orb_slice = disk_inner_stable_circ_orb_arr.as_slice().unwrap();
    let disk_surf_slice = disk_surf_arr.as_slice().unwrap();
    let disk_radius_outer_slice = disk_radius_outer_arr.as_slice().unwrap();
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
        .zip(disk_inner_stable_circ_orb_slice)
        .zip(disk_surf_slice)
        .zip(disk_radius_outer_slice)
        .zip(rng_slice)
        .enumerate()
        .for_each(|(i, ((((((((disk_bh_retro_masses, disk_bh_retro_orbs_a), disk_bh_retro_orbs_ecc), disk_bh_retro_orbs_inc), disk_bh_retro_arg_periapse), disk_inner_stable_circ_orb), disk_surf), disk_radius_outer), rng))| {

        let cos_pm1_mask: bool = disk_bh_retro_arg_periapse.cos().abs() >= 0.5;
        let cos_0_mask: bool = !cos_pm1_mask;

        // these two are related but not exhaustive
        let no_max_ecc_retro_mask: bool = (disk_bh_retro_orbs_ecc < &STEP2_ECC_0) & (disk_bh_retro_orbs_inc.abs() >= PI/2.0); 
        let max_ecc_mask: bool = disk_bh_retro_orbs_ecc >= &STEP2_ECC_0;
        let barely_prograde_mask: bool = disk_bh_retro_orbs_inc.abs() < PI/2.0;
        let ecc_unreliable_mask: bool = !(no_max_ecc_retro_mask | max_ecc_mask | barely_prograde_mask);
        if ecc_unreliable_mask {panic!("ECC Warning: retrograde orbital parameters out of range, behavior unreliable")};

        let condition1 = cos_pm1_mask & no_max_ecc_retro_mask;
        let condition2 = cos_pm1_mask & max_ecc_mask;
        let condition3 = cos_pm1_mask & barely_prograde_mask;
        let conditionw0 = cos_0_mask;

        let periapse = if cos_pm1_mask {PERIAPSE_1} else if cos_0_mask {PERIAPSE_0} else {-100.5};

        // try to convert this into an exhaustive match
        let (semi_maj_0, ecc0, inc0) = if condition1 {
            (STEP1_SEMI_MAJ_0, STEP1_ECC_0, STEP1_INC_0)
        } else if condition2 {
            (STEP2_SEMI_MAJ_0, STEP2_ECC_0, STEP2_INC_0)
        } else if condition3 {
            (STEP3_SEMI_MAJ_0, STEP3_ECC_0, STEP3_INC_0)
        } else if conditionw0 {
            (STEPW0_SEMI_MAJ_0, STEPW0_ECC_0, STEPW0_INC_0)
        } else { (-100.5, -100.5, -100.5) };

        // check that this is the right input
        let semi_maj_axis = si_from_r_g(smbh_mass, *disk_bh_retro_orbs_a);

        let (tau_e_current, tau_a_current) = tau_ecc_dyn_local(smbh_mass, *disk_bh_retro_masses, *disk_bh_retro_orbs_ecc, *disk_bh_retro_orbs_inc, *disk_bh_retro_arg_periapse, *disk_surf, semi_maj_axis);
        let tau_inc_current = tau_inc_dyn_local(smbh_mass, *disk_bh_retro_masses, *disk_bh_retro_orbs_ecc, *disk_bh_retro_orbs_inc, *disk_bh_retro_arg_periapse, *disk_surf, semi_maj_axis);

        let (tau_e_ref, tau_a_ref) = tau_ecc_dyn_local(SMBH_MASS_0, semi_maj_0, ecc0, inc0, periapse, *disk_surf, semi_maj_axis);
        let tau_inc_ref = tau_inc_dyn_local(SMBH_MASS_0, semi_maj_0, ecc0, inc0, periapse, *disk_surf, semi_maj_axis);

        let tau_e_div = tau_e_current / tau_e_ref;
        let tau_a_div = tau_a_current / tau_a_ref;
        let tau_inc_div = tau_inc_current / tau_inc_ref;

        // let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = if condition1 {
        //     (STEP1_TIME * tau_e_div, STEP1_TIME * tau_a_div, STEP1_TIME * tau_inc_div, )
        // } else if condition2 {
        //     (STEP2_TIME * tau_e_div, STEP2_TIME * tau_a_div, STEP2_TIME * tau_inc_div, )
        // } else if condition3 {
        //     (STEP3_TIME * tau_e_div, STEP3_TIME * tau_a_div, STEP3_TIME * tau_inc_div, )
        // } else if conditionw0 {
        //     (STEPW0_TIME * tau_e_div, STEPW0_TIME * tau_a_div, STEPW0_TIME * tau_inc_div, )
        // } else {
        //     (-100.5, -100.5, -100.5)
        // };


        if condition1 {
            let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = (STEP1_TIME * tau_e_div, STEP1_TIME * tau_a_div, STEP1_TIME * tau_inc_div);
            let disk_bh_retro_orbs_ecc_new: f64 = (
                disk_bh_retro_orbs_ecc * (
                    1.0 + STEP1_DELTA_ECC / disk_bh_retro_orbs_ecc * (timestep_duration_yr / ecc_scale_factor))
                ).clamp(0.0, 1.0-EPSILON);
            let disk_bh_retro_orbs_a_new: f64 = (
                disk_bh_retro_orbs_a * (
                    1.0 - STEP1_DELTA_SEMIMAJ / disk_bh_retro_orbs_a * (timestep_duration_yr / semimaj_scale_factor)
                )).clamp(*disk_inner_stable_circ_orb, f64::INFINITY);
            let disk_bh_retro_orbs_inc_new: f64 = (
                disk_bh_retro_orbs_inc * (
                    1.0 - STEP1_DELTA_INC / disk_bh_retro_orbs_inc * (timestep_duration_yr / inc_scale_factor)
                )).clamp(0.0, f64::INFINITY);

            // properly speaking, this could be part of the clamp operation, but the behavior is
            // slightly different
            // disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new > *disk_radius_outer] = disk_radius_outer - epsilon_orb_a[disk_bh_retro_orbs_a_new > *disk_radius_outer]
            let disk_bh_retro_orbs_a_new = if disk_bh_retro_orbs_a_new > *disk_radius_outer {
                let epsilon_orb_a = disk_radius_outer * 
                    ((disk_bh_retro_masses / (3.0 * (disk_bh_retro_masses + smbh_mass))).cbrt()) * rng;
                disk_radius_outer - epsilon_orb_a
            } else {
                disk_bh_retro_orbs_a_new
            };

            out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
            out_a_slice[i] = disk_bh_retro_orbs_a_new;
            out_inc_slice[i] = disk_bh_retro_orbs_inc_new;
        } else if condition2 { 
            let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = (STEP2_TIME * tau_e_div, STEP2_TIME * tau_a_div, STEP2_TIME * tau_inc_div);
            let disk_bh_retro_orbs_ecc_new: f64 = (disk_bh_retro_orbs_ecc * (1.0 + STEP2_DELTA_ECC / disk_bh_retro_orbs_ecc * (timestep_duration_yr / ecc_scale_factor))).clamp(0.0, 1.0-EPSILON);
            let disk_bh_retro_orbs_a_new: f64 = (
                disk_bh_retro_orbs_a * (
                    1.0 - STEP2_DELTA_SEMIMAJ / disk_bh_retro_orbs_a * (timestep_duration_yr / semimaj_scale_factor)
                )).clamp(*disk_inner_stable_circ_orb, f64::INFINITY);
            let disk_bh_retro_orbs_inc_new: f64 = (
                disk_bh_retro_orbs_inc * (
                    1.0 - STEP2_DELTA_INC / disk_bh_retro_orbs_inc * (timestep_duration_yr / inc_scale_factor)
                )).clamp(0.0, f64::INFINITY);

            let disk_bh_retro_orbs_a_new = if disk_bh_retro_orbs_a_new > *disk_radius_outer {
                let epsilon_orb_a = disk_radius_outer * 
                    ((disk_bh_retro_masses / (3.0 * (disk_bh_retro_masses + smbh_mass))).cbrt()) * rng;
                disk_radius_outer - epsilon_orb_a
            } else {
                disk_bh_retro_orbs_a_new
            };

            out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
            out_a_slice[i] = disk_bh_retro_orbs_a_new;
            out_inc_slice[i] = disk_bh_retro_orbs_inc_new;
        } else if condition3 {
            let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = (STEP3_TIME * tau_e_div, STEP3_TIME * tau_a_div, STEP3_TIME * tau_inc_div);
            let disk_bh_retro_orbs_ecc_new: f64 = (disk_bh_retro_orbs_ecc * (1.0 + STEP3_DELTA_ECC / disk_bh_retro_orbs_ecc * (timestep_duration_yr / ecc_scale_factor))).clamp(0.0, 1.0-EPSILON);
            let disk_bh_retro_orbs_a_new: f64 = (
                disk_bh_retro_orbs_a * (
                    1.0 - STEP3_DELTA_SEMIMAJ / disk_bh_retro_orbs_a * (timestep_duration_yr / semimaj_scale_factor)
                )).clamp(*disk_inner_stable_circ_orb, f64::INFINITY);
            let disk_bh_retro_orbs_inc_new: f64 = (
                disk_bh_retro_orbs_inc * (
                    1.0 - STEP3_DELTA_INC / disk_bh_retro_orbs_inc * (timestep_duration_yr / inc_scale_factor)
                )).clamp(0.0, f64::INFINITY);

            let disk_bh_retro_orbs_a_new = if disk_bh_retro_orbs_a_new > *disk_radius_outer {
                let epsilon_orb_a = disk_radius_outer * 
                    ((disk_bh_retro_masses / (3.0 * (disk_bh_retro_masses + smbh_mass))).cbrt()) * rng;
                disk_radius_outer - epsilon_orb_a
            } else {
                disk_bh_retro_orbs_a_new
            };
            out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
            out_a_slice[i] = disk_bh_retro_orbs_a_new;
            out_inc_slice[i] = disk_bh_retro_orbs_inc_new;

        } else if conditionw0 {
            let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = (STEPW0_TIME * tau_e_div, STEPW0_TIME * tau_a_div, STEPW0_TIME * tau_inc_div);
            let disk_bh_retro_orbs_ecc_new: f64 = (disk_bh_retro_orbs_ecc * (1.0 + STEPW0_DELTA_ECC / disk_bh_retro_orbs_ecc * (timestep_duration_yr / ecc_scale_factor))).clamp(0.0, 1.0-EPSILON);
            let disk_bh_retro_orbs_a_new: f64 = (
                disk_bh_retro_orbs_a * (
                    1.0 - STEPW0_DELTA_SEMIMAJ / disk_bh_retro_orbs_a * (timestep_duration_yr / semimaj_scale_factor)
                )).clamp(*disk_inner_stable_circ_orb, f64::INFINITY);
            let disk_bh_retro_orbs_inc_new: f64 = (
                disk_bh_retro_orbs_inc * (
                    1.0 - STEPW0_DELTA_INC / disk_bh_retro_orbs_inc * (timestep_duration_yr / inc_scale_factor)
                )).clamp(0.0, f64::INFINITY);

            let disk_bh_retro_orbs_a_new = if disk_bh_retro_orbs_a_new > *disk_radius_outer {
                let epsilon_orb_a = disk_radius_outer * 
                    ((disk_bh_retro_masses / (3.0 * (disk_bh_retro_masses + smbh_mass))).cbrt()) * rng;
                disk_radius_outer - epsilon_orb_a
            } else {
                disk_bh_retro_orbs_a_new
            };

            out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
            out_a_slice[i] = disk_bh_retro_orbs_a_new;
            out_inc_slice[i] = disk_bh_retro_orbs_inc_new;
        } else {
            // are these actually totally unnecessary in this branch????
            // todo: double check this
            // let (ecc_scale_factor, semimaj_scale_factor, inc_scale_factor) = (-100.5, -100.5, -100.5);

            let disk_bh_retro_orbs_ecc_new: f64 = 0.0;   
            let disk_bh_retro_orbs_a_new: f64 = 0.0;
            let disk_bh_retro_orbs_inc_new: f64 = 0.0;

            out_ecc_slice[i] = disk_bh_retro_orbs_ecc_new;
            out_a_slice[i] = disk_bh_retro_orbs_a_new;
            out_inc_slice[i] = disk_bh_retro_orbs_inc_new;
        }

     // todo: which of these finity checks are actually necessary?? 


        //     # Check Finite
        //     nan_mask = (
        //         ~np.isfinite(disk_bh_retro_orbs_ecc_new) | \
        //         ~np.isfinite(disk_bh_retro_orbs_a_new) | \
        //         ~np.isfinite(disk_bh_retro_orbs_inc_new) \
        //     )
        //     if np.sum(nan_mask) > 0:
        //         # Check for objects inside 12.1 R_g
        //         if all(disk_bh_retro_orbs_a[nan_mask] < 12.1):
        //             disk_bh_retro_orbs_ecc_new[nan_mask] = disk_bh_retro_orbs_ecc[nan_mask]
        //             # Inside ACTUAL ISCO; might get caught better
        //             disk_bh_retro_orbs_a_new[nan_mask] = 5.9
        //             # It's been eaten
        //             disk_bh_retro_orbs_inc_new[nan_mask] = 0.
        //         else:
        //             print("nan_mask:",np.where(nan_mask))
        //             print("nan old ecc:",disk_bh_retro_orbs_ecc[nan_mask])
        //             print("disk_bh_retro_masses:", disk_bh_retro_masses[nan_mask])
        //             print("disk_bh_retro_orbs_a:", disk_bh_retro_orbs_a[nan_mask])
        //             print("disk_bh_retro_orbs_inc:", disk_bh_retro_orbs_inc[nan_mask])
        //             print("disk_bh_retro_arg_periapse:", disk_bh_retro_arg_periapse[nan_mask])
        //             disk_bh_retro_orbs_ecc_new[nan_mask] = 2.
        //             disk_bh_retro_orbs_a_new[nan_mask] = 0.
        //             disk_bh_retro_orbs_inc_new[nan_mask] = 0.
        //             raise RuntimeError("Finite check failed for disk_bh_retro_orbs_ecc_new")
        //
        //     assert np.all(disk_bh_retro_orbs_a_new < disk_radius_outer), \
        //         "disk_bh_retro_orbs_a_new has values greater than disk_radius_outer"
        //     assert np.all(disk_bh_retro_orbs_a_new >= 0), \
        //         "disk_bh_retro_orbs_a_new has values < 0"
        //
        //     return disk_bh_retro_orbs_ecc_new, disk_bh_retro_orbs_a_new, disk_bh_retro_orbs_inc_new
    });
    // todo: collect or manually append into the pre-set arrays

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
//
//     Notes
//     -----
//     To avoid having to install and couple to SpaceHub, and run N-body code
//     this is a distinctly half-assed treatment of retrograde orbiters, based
//     LOOSELY on Wang, Zhu & Lin 2024 (WZL). Evolving all orbital params simultaneously.
//     Using lots of if statements to pretend we're interpolating.
//     Hardcoding some stuff from WZL figs 7, 8 & 12 (see comments).
//     Arg of periapse = w in comments below
//
//     """
//
//     # first handle cos(w)=+/-1 (assume abs(cos(w))>0.5)
//     #   this evolution is multistage:
//     #       1. radialize, semimaj axis shrinks (slower), flip (very slowly)
//     #       2. flip (very fast), circ (slowly), constant semimaj axis
//     #       3. i->0.000 (very fast), circ & shrink semimaj axis slightly slower
//     #
//     #      For smbh_mass=1e8Msun, orbiter_mass=30Msun, SG disk surf dens (fig 12 WZL)
//     #       1. in 1.5e5yrs e=0.7->0.9999 (roughly), a=100rg->60rg, i=175->165deg
//     #       2. in 1e4yrs i=165->12deg, e=0.9999->0.9, a=60rg
//     #       3. in 1e4yrs i=12->0.0deg, e=0.9->0.5, a=60->20rg
//
//     # Housekeeping: if you're too close to ecc=1.0, nothing works, so
//     epsilon = 1.0e-8
//
//     # hardcoded awfulness coming up:
//     smbh_mass_0 = 1e8  # solar masses, for scaling
//     orbiter_mass_0 = 30.0  # solar masses
//     periapse_1 = 0.0  # radians
//     periapse_0 = np.pi / 2.0  # radians
//
//     step1_ecc_0 = 0.7
//     step1_inc_0 = np.pi * (175.0 / 180.0)  # rad
//     step1_semi_maj_0 = 100.0  # r_g
//
//     step2_ecc_0 = 0.9999
//     step2_inc_0 = np.pi * (165.0 / 180.0)  # rad
//     step2_semi_maj_0 = 60.0  # r_g
//
//     step3_ecc_0 = 0.9
//     step3_inc_0 = np.pi * (12.0 / 180.0)  # rad
//     step3_semi_maj_0 = step2_semi_maj_0  # r_g
//
//     step3_ecc_f = 0.5
//     step3_inc_f = 0.0  # rad
//     step3_semi_maj_f = 20.0  # r_g
//
//     stepw0_ecc_0 = 0.7
//     stepw0_inc_0 = np.pi * (175.0 / 180.0)  # rad
//     stepw0_semi_maj_0 = 100.0  # r_g
//
//     stepw0_ecc_f = 0.5
//     stepw0_inc_f = np.pi * (170.0 / 180.0)  # rad
//     stepw0_semi_maj_f = 60.0  # r_g
//
//     step1_time = 1.5e5  # years
//     step1_delta_ecc = step2_ecc_0 - step1_ecc_0
//     step1_delta_semimaj = step1_semi_maj_0 - step2_semi_maj_0  #rg
//     step1_delta_inc = step1_inc_0 - step2_inc_0  #rad
//
//     step2_time = 1.4e4  # years
//     step2_delta_ecc = step2_ecc_0 - step3_ecc_0
//     step2_delta_semimaj = step2_semi_maj_0 - step3_semi_maj_0  #rg
//     step2_delta_inc = step2_inc_0 - step3_inc_0
//
//     step3_time = 1.4e4  # years
//     step3_delta_ecc = step3_ecc_0 - step3_ecc_f
//     step3_delta_semimaj = step3_semi_maj_0 - step3_semi_maj_f  #rg
//     step3_delta_inc = step3_inc_0 - step3_inc_f
//
//     # Then figure out cos(w)=0
//     # this evolution does one thing: shrink semimaj axis, circ (slowly), flip (even slower)
//     #   scaling from fig 8 WZL comparing cos(w)=0 to cos(w)=+/-1
//     #       tau_semimaj~1/100, tau_ecc~1/1000, tau_inc~1/5000
//     #       at high inc, large ecc
//     #
//     #      Estimating for smbh_mass=1e8Msun, orbiter_mass=30Msun, SG disk surf dens
//     #       in 1.5e7yrs a=100rg->60rg, e=0.7->0.5, i=175->170deg
//     stepw0_time = 1.5e7  # years
//     stepw0_delta_ecc = stepw0_ecc_0 - stepw0_ecc_f
//     stepw0_delta_semimaj = stepw0_semi_maj_0 - stepw0_semi_maj_f  #rg
//     stepw0_delta_inc = stepw0_inc_0 - stepw0_inc_f
//
//     # setup output arrays
//     disk_bh_retro_orbs_ecc_new = np.zeros(len(disk_bh_retro_orbs_ecc))
//     disk_bh_retro_orbs_inc_new = np.zeros(len(disk_bh_retro_orbs_inc))
//     disk_bh_retro_orbs_a_new = np.zeros(len(disk_bh_retro_orbs_a))
//
//     tau_e_current = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     tau_a_current = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     tau_e_ref = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     tau_a_ref = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     ecc_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     semimaj_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     inc_scale_factor = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//
//     # cosine masks
//     # returns True for values where cos(w)~+/-1
//     cos_pm1_mask = np.abs(np.cos(disk_bh_retro_arg_periapse)) >= 0.5
//     # returns True for values where cos(w)=0 (assume abs(cos(w))<0.5)
//     cos_0_mask = np.abs(np.cos(disk_bh_retro_arg_periapse)) < 0.5
//     cos_unreliable_mask = ~(cos_pm1_mask | cos_0_mask)
//     if cos_unreliable_mask.sum() > 0:
//         print("COS Warning: retrograde orbital parameters out of range, behavior unreliable")
//
//     # eccentricity/inclination masks
//     # returns True for values where we haven't hit our max ecc for step 1, and remain somewhat retrograde
//     no_max_ecc_retro_mask = (disk_bh_retro_orbs_ecc < step2_ecc_0) & (np.abs(disk_bh_retro_orbs_inc) >= np.pi / 2.0)
//     # returns True for values where we have hit max ecc, which sends us to step 2
//     max_ecc_mask = disk_bh_retro_orbs_ecc >= step2_ecc_0
//     # returns True for values where our inc is even barely prograde... hopefully this works ok...
//     # this should work as long as we're only tracking stuff originally retrograde
//     barely_prograde_mask = np.abs(disk_bh_retro_orbs_inc) < (np.pi / 2.0)
//     ecc_unreliable_mask = ~(no_max_ecc_retro_mask | max_ecc_mask | barely_prograde_mask)
//     if ecc_unreliable_mask.sum() > 0:
//         print("ECC Warning: retrograde orbital parameters out of range, behavior unreliable")
//
//     # Set up arrays for hardcoded values
//     semi_maj_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     ecc_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     inc_0 = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//     periapse = np.full(disk_bh_retro_arg_periapse.size, -100.5)
//
//     # Fill with values
//     semi_maj_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_semi_maj_0
//     semi_maj_0[cos_pm1_mask & max_ecc_mask] = step2_semi_maj_0
//     semi_maj_0[cos_pm1_mask & barely_prograde_mask] = step3_semi_maj_0
//     semi_maj_0[cos_0_mask] = stepw0_semi_maj_0
//
//     ecc_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_ecc_0
//     ecc_0[cos_pm1_mask & max_ecc_mask] = step2_ecc_0
//     ecc_0[cos_pm1_mask & barely_prograde_mask] = step3_ecc_0
//     ecc_0[cos_0_mask] = stepw0_ecc_0
//
//     inc_0[cos_pm1_mask & no_max_ecc_retro_mask] = step1_inc_0
//     inc_0[cos_pm1_mask & max_ecc_mask] = step2_inc_0
//     inc_0[cos_pm1_mask & barely_prograde_mask] = step3_inc_0
//     inc_0[cos_0_mask] = stepw0_inc_0
//
//     periapse[cos_pm1_mask] = periapse_1
//     periapse[cos_0_mask] = periapse_0
//
//     # Get current tau values
//     # tau_e_current_orig, tau_a_current_orig = tau_ecc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
//     #                                            disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc,
//     #                                            disk_surf_density_func, r_g_in_meters)
//     tau_e_current, tau_a_current = tau_ecc_dyn_optimized(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
//                                                disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc, disk_bh_retro_orbs_inc,
//                                                disk_surf_density_func, r_g_in_meters)
//     # assert(np.allclose(tau_a_current, tau_a_current_orig))
//     # assert(np.allclose(tau_e_current, tau_e_current_orig))
//
//     # tau_inc_current_orig = tau_inc_dyn(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
//     #                               disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc,
//     #                               disk_bh_retro_orbs_inc, disk_surf_density_func, r_g_in_meters)
//     tau_inc_current = tau_inc_dyn_optimized(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses,
//                                   disk_bh_retro_arg_periapse, disk_bh_retro_orbs_ecc,
//                                   disk_bh_retro_orbs_inc, disk_surf_density_func, r_g_in_meters)
//     # assert(np.allclose(tau_inc_current, tau_inc_current_orig))
//
//     # Get reference tau values
//     # tau_e_ref_orig, tau_a_ref_orig = tau_ecc_dyn(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func, r_g_in_meters)
//     tau_e_ref, tau_a_ref = tau_ecc_dyn_optimized(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func, r_g_in_meters)
//     # assert(np.allclose(tau_e_ref, tau_e_ref_orig))
//     # assert(np.allclose(tau_a_ref, tau_a_ref_orig))
//
//     # tau_inc_ref_orig = tau_inc_dyn(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func, r_g_in_meters)
//     tau_inc_ref = tau_inc_dyn_optimized(smbh_mass_0, semi_maj_0, orbiter_mass_0, periapse, ecc_0, inc_0, disk_surf_density_func, r_g_in_meters)
//     # assert(np.allclose(tau_inc_ref, tau_inc_ref_orig))
//
//     if (tau_e_current == -100.5).sum() > 0:
//         print("TAU Warning: retrograde orbital parameters out of range, behavior unreliable")
//
//     # Get ecc scale factors
//     tau_e_div = tau_e_current / tau_e_ref
//     ecc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_e_div[cos_pm1_mask & no_max_ecc_retro_mask]
//     ecc_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_e_div[cos_pm1_mask & max_ecc_mask]
//     ecc_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_e_div[cos_pm1_mask & barely_prograde_mask]
//     ecc_scale_factor[cos_0_mask] = stepw0_time * tau_e_div[cos_0_mask]
//     # Get semimaj scale factors
//     tau_a_div = tau_a_current / tau_a_ref
//     semimaj_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_a_div[cos_pm1_mask & no_max_ecc_retro_mask]
//     semimaj_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_a_div[cos_pm1_mask & max_ecc_mask]
//     semimaj_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_a_div[cos_pm1_mask & barely_prograde_mask]
//     semimaj_scale_factor[cos_0_mask] = stepw0_time * tau_a_div[cos_0_mask]
//     # Get inc scale factors
//     tau_inc_div = tau_inc_current / tau_inc_ref
//     inc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask] = step1_time * tau_inc_div[cos_pm1_mask & no_max_ecc_retro_mask]
//     inc_scale_factor[cos_pm1_mask & max_ecc_mask] = step2_time * tau_inc_div[cos_pm1_mask & max_ecc_mask]
//     inc_scale_factor[cos_pm1_mask & barely_prograde_mask] = step3_time * tau_inc_div[cos_pm1_mask & barely_prograde_mask]
//     inc_scale_factor[cos_0_mask] = stepw0_time * tau_inc_div[cos_0_mask]
//
//     # Calculate new orb_ecc values
//     disk_bh_retro_orbs_ecc_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & no_max_ecc_retro_mask] * (
//         1.0 + step1_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
//     disk_bh_retro_orbs_ecc_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & max_ecc_mask] * (
//         1.0 - step2_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & max_ecc_mask]))
//     disk_bh_retro_orbs_ecc_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_ecc[cos_pm1_mask & barely_prograde_mask] * (
//         1.0 - step3_delta_ecc / disk_bh_retro_orbs_ecc[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / ecc_scale_factor[cos_pm1_mask & barely_prograde_mask]))
//     disk_bh_retro_orbs_ecc_new[cos_0_mask] = disk_bh_retro_orbs_ecc[cos_0_mask] * (
//         1.0 - stepw0_delta_ecc / disk_bh_retro_orbs_ecc[cos_0_mask] * (timestep_duration_yr / ecc_scale_factor[cos_0_mask]))
//
//     # Calculate new orb_a values
//     disk_bh_retro_orbs_a_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & no_max_ecc_retro_mask] * (
//         1.0 - step1_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
//     disk_bh_retro_orbs_a_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & max_ecc_mask] * (
//         1.0 - step2_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & max_ecc_mask]))
//     disk_bh_retro_orbs_a_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_a[cos_pm1_mask & barely_prograde_mask] * (
//         1.0 - step3_delta_semimaj / disk_bh_retro_orbs_a[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_pm1_mask & barely_prograde_mask]))
//     disk_bh_retro_orbs_a_new[cos_0_mask] = disk_bh_retro_orbs_a[cos_0_mask] * (
//         1.0 - stepw0_delta_semimaj / disk_bh_retro_orbs_a[cos_0_mask] * (timestep_duration_yr / semimaj_scale_factor[cos_0_mask]))
//
//     # Calculate new orb_inc values
//     disk_bh_retro_orbs_inc_new[cos_pm1_mask & no_max_ecc_retro_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & no_max_ecc_retro_mask] * (
//         1.0 - step1_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & no_max_ecc_retro_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & no_max_ecc_retro_mask]))
//     disk_bh_retro_orbs_inc_new[cos_pm1_mask & max_ecc_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & max_ecc_mask] * (
//         1.0 - step2_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & max_ecc_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & max_ecc_mask]))
//     disk_bh_retro_orbs_inc_new[cos_pm1_mask & barely_prograde_mask] = disk_bh_retro_orbs_inc[cos_pm1_mask & barely_prograde_mask] * (
//         1.0 - step3_delta_inc / disk_bh_retro_orbs_inc[cos_pm1_mask & barely_prograde_mask] * (timestep_duration_yr / inc_scale_factor[cos_pm1_mask & barely_prograde_mask]))
//     disk_bh_retro_orbs_inc_new[cos_0_mask] = disk_bh_retro_orbs_inc[cos_0_mask] * (
//         1.0 - stepw0_delta_inc / disk_bh_retro_orbs_inc[cos_0_mask] * (timestep_duration_yr / inc_scale_factor[cos_0_mask]))
//
//     # Catch overshooting ecc = 0
//     disk_bh_retro_orbs_ecc_new[disk_bh_retro_orbs_ecc_new < 0.0] = 0.0
//     # catch overshooting ecc=1, actually eqns not appropriate for ecc=1.0
//     disk_bh_retro_orbs_ecc_new[disk_bh_retro_orbs_ecc_new >= 1.0 - epsilon] = 1.0 - epsilon
//     # Catch overshooting semi-major axis, set to disk_inner_stable_circ_orb
//     disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new <= 0.0] = disk_inner_stable_circ_orb
//     # Catch overshooting inc, set to 0.0
//     disk_bh_retro_orbs_inc_new[disk_bh_retro_orbs_inc_new <= 0.0] = 0.0
//
//     # Check Finite
//     nan_mask = (
//         ~np.isfinite(disk_bh_retro_orbs_ecc_new) | \
//         ~np.isfinite(disk_bh_retro_orbs_a_new) | \
//         ~np.isfinite(disk_bh_retro_orbs_inc_new) \
//     )
//     if np.sum(nan_mask) > 0:
//         # Check for objects inside 12.1 R_g
//         if all(disk_bh_retro_orbs_a[nan_mask] < 12.1):
//             disk_bh_retro_orbs_ecc_new[nan_mask] = disk_bh_retro_orbs_ecc[nan_mask]
//             # Inside ACTUAL ISCO; might get caught better
//             disk_bh_retro_orbs_a_new[nan_mask] = 5.9
//             # It's been eaten
//             disk_bh_retro_orbs_inc_new[nan_mask] = 0.
//         else:
//             print("nan_mask:",np.where(nan_mask))
//             print("nan old ecc:",disk_bh_retro_orbs_ecc[nan_mask])
//             print("disk_bh_retro_masses:", disk_bh_retro_masses[nan_mask])
//             print("disk_bh_retro_orbs_a:", disk_bh_retro_orbs_a[nan_mask])
//             print("disk_bh_retro_orbs_inc:", disk_bh_retro_orbs_inc[nan_mask])
//             print("disk_bh_retro_arg_periapse:", disk_bh_retro_arg_periapse[nan_mask])
//             disk_bh_retro_orbs_ecc_new[nan_mask] = 2.
//             disk_bh_retro_orbs_a_new[nan_mask] = 0.
//             disk_bh_retro_orbs_inc_new[nan_mask] = 0.
//             raise RuntimeError("Finite check failed for disk_bh_retro_orbs_ecc_new")
//
//     # Anything outside the disk is brought back in
//     # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
//     epsilon_orb_a = disk_radius_outer * ((disk_bh_retro_masses / (3 * (disk_bh_retro_masses + smbh_mass)))**(1. / 3.)) * rng.uniform(size=len(disk_bh_retro_masses))
//     disk_bh_retro_orbs_a_new[disk_bh_retro_orbs_a_new > disk_radius_outer] = disk_radius_outer - epsilon_orb_a[disk_bh_retro_orbs_a_new > disk_radius_outer]
//
//     assert np.all(disk_bh_retro_orbs_a_new < disk_radius_outer), \
//         "disk_bh_retro_orbs_a_new has values greater than disk_radius_outer"
//     assert np.all(disk_bh_retro_orbs_a_new >= 0), \
//         "disk_bh_retro_orbs_a_new has values < 0"
//
//     return disk_bh_retro_orbs_ecc_new, disk_bh_retro_orbs_a_new, disk_bh_retro_orbs_inc_new
