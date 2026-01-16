//! Some functions to implement faster cube rooting
//!

use pyo3::prelude::*;
use crate::constants::G;

use std::f64::consts::PI;

// in order to avoid heap allocations each time we run this function, we can instead return a
// fixed-size array and just note how many of the roots (either 1 or 3) are valid

/// A Rust implementation of a dedicated cubic root solver for McFACTS
#[pyfunction(signature = (x0, y0))]
pub fn cubic_y_root_cardano(x0: f64, y0: f64) -> ([f64; 3], usize) {
    if x0 == 0.0 {
        ([y0/1.5, 0.0, 0.0], 1) //padding out the array
    } else {
        let p = 1.5/x0;
        let q  = -y0/x0;

        let delta = (q/2.0).powi(2) + (p/3.0).powi(3);

        if delta > 0.0 {
            // one real root path
            let sqrt_delta = delta.sqrt();
            let u = (-q/2.0 + sqrt_delta).cbrt();
            let v = (-q/2.0 - sqrt_delta).cbrt();
            ([u + v, 0.0, 0.0], 1) //padding out the array
        } else {
            // three real roots path
            let term1 = 2.0 * (-p / 3.0).sqrt();
            let phi = ((3.0 * q) / (p * term1)).acos();

            let y1 = term1 * (phi / 3.0).cos();
            let y2 = term1 * ((phi + 2.0 * PI) / 3.0).cos();
            let y3 = term1 * ((phi + 4.0 * PI) / 3.0).cos();
            ([y1, y2, y3], 3)
        }
    }
}

#[pyfunction(signature=(x0, y0, omega_s))] 
pub fn cubic_finite_step_root_cardano(
    x0: f64,
    y0: f64,
    omega_s: f64
) -> ([[f64; 2]; 3], usize) {
    let p = 2.0 * (x0 - (omega_s * y0));
    let q = 2.0 * omega_s;

    // This block determines the y-roots using Cardano's method.
    // This logic appears to correctly mirror your Python version.
    let (roots_y_array, y_count) = if p == 0.0 {
        ([(-q).cbrt(), 0.0, 0.0], 1)
    } else {
        let delta = (q / 2.0).powi(2) + (p / 3.0).powi(3);

        if delta >= 0.0 {
            // One real root path
            let sqrt_delta = delta.sqrt();
            let u = (-q / 2.0 + sqrt_delta).cbrt();
            let v = (-q / 2.0 - sqrt_delta).cbrt();
            ([u + v, 0.0, 0.0], 1)
        } else {
            // Three real roots path
            let term1 = 2.0 * (-p / 3.0).sqrt();
            // Note: Add clamping here to prevent panics from float errors
            // if the argument is slightly outside [-1, 1]
            let acos_arg = ((3.0 * q) / (p * term1)).clamp(-1.0, 1.0); 
            let phi = acos_arg.acos();

            let y1 = term1 * (phi / 3.0).cos();
            let y2 = term1 * ((phi + 2.0 * PI) / 3.0).cos();
            let y3 = term1 * ((phi + 4.0 * PI) / 3.0).cos();
            ([y1, y2, y3], 3)
        }
    };

    let mut buffer = [[0.0; 2]; 3];
    let mut pair_count = 0;
    
    // iterate through the valid y-roots found above
    for y_root in roots_y_array.iter().take(y_count) {

        // calculate the corresponding x-root before filtering
        let x_root = -1.0 / (2.0 * y_root.powi(2));

        // apply all three conditions from the Python `indx_ok` mask
        let is_valid_pair = *y_root > 0.0 && x_root.is_finite() && x_root < 0.0;

        if is_valid_pair {
            // ensure we don't write past the end of our 3-element buffer
            if pair_count < 3 {
                buffer[pair_count] = [x_root, *y_root];
                pair_count += 1;
            } else {
                // this should be unreachable if y_count is max 3,
                // but it's good practice for safety
                break;
            }
        }
    }

    (buffer, pair_count)
}

fn components_from_el(e: f64, l:f64, units: Option<&str>, smbh_mass: f64) -> (f64, f64) {

    let g_val = if units == Some("geometric") {
        G
    } else {
        1.0
    };

    let g_times_mass = g_val * smbh_mass;
    
    let orb_a = - g_times_mass/(2.0*e);

    let mut one_minus_ecc2_sqrt = l/(g_times_mass * orb_a).sqrt();

    one_minus_ecc2_sqrt = if (one_minus_ecc2_sqrt - 1.0 > 0.0) && (one_minus_ecc2_sqrt - 1.0 < 1e-2) {
        1.0-1e-2
    } else {
        one_minus_ecc2_sqrt
    };

    if one_minus_ecc2_sqrt > 1.0 {
        panic!("Impossible eccentricity value")
    } 

    let ecc = (1.0 - one_minus_ecc2_sqrt.powi(2)).sqrt();

    (orb_a / (2.0 * smbh_mass), ecc)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (e1, l1, e2, l2, delta_e, m1, m2, units = None, smbh_mass = None))]
pub fn transition_physical_as_el(
    e1: f64, 
    l1: f64,
    e2: f64,
    l2: f64,
    delta_e: f64,
    m1: f64,
    m2: f64,
    units: Option<&str>, // defaults to 'geometric'
    smbh_mass: Option<f64>, // defaults to 10e8
) -> (f64, f64, f64, f64) {

    let units = units.unwrap_or("geometric");

    let smbh_mass = smbh_mass.unwrap_or(10.0e8);

    let g_val = if units == "geometric" {
        1.0
    } else {
        G
    };

    // Assume consistent units SI only
    let eps1 = e1/m1;
    let eps2 = e2/m2;
    let ell1 = l1/m1;
    let ell2 = l2/m2;

    // find omega_0 scale, which is based on the acceptor (2) non-eccentric object. This means ell0 = ell2
    let ell0 = ell2;
    let g_times_mass = g_val * smbh_mass;
    let omega_0 = g_times_mass.powi(2) / ell0.powi(3);
    let eps0 = ell0 * omega_0;

    let base = -2.0 * eps2;
    let omega_2 = base.powf(1.5) / g_times_mass;
    // let omega_2 = (-2.0 * eps2).powi(3) / (g_times_mass);

    let x0 = eps2/eps0;
    let y0 = 1.0; // ell2/ell0, which is 1 by construction

    let x0_alt = eps1/eps0;
    let y0_alt = ell1/ell0;
    
    if delta_e * (e2 - e1) < 0.0 {
        let omega_trial = omega_2;

            // This is the new logic block that replaces your snippet.
        // We calculate the final delta_e inside a block and assign it.
        let delta_e = {
            // 1. Get the 2D array of roots and the number of valid pairs.
            let (stepsize_roots_arr, stepsize_count) = cubic_finite_step_root_cardano(
                x0_alt,
                y0_alt,
                omega_trial / omega_0,
            );

            // 2. Get a slice containing only the valid root pairs.
            let valid_roots = &stepsize_roots_arr[..stepsize_count];

            // 3. Find the *first* pair where the x-value (at index 0) is less than x0.
            //    `.find()` is efficient because it stops searching as soon as it finds a match.
            let first_matching_pair = valid_roots
                .iter()
                .find(|root_pair| root_pair[0] < x0);

            // 4. Check if we found a matching pair. `first_matching_pair` is an `Option`.
            if let Some(matching_pair) = first_matching_pair {
                // We found one. `matching_pair` is `&[x, y]`.
                let x_root = matching_pair[0];
                let delta_e_max = m1 * (x_root - x0_alt) * eps0;

                // 5. Apply the cap, similar to the original logic.
                if delta_e.abs() > delta_e_max.abs() {
                    delta_e_max // This value will be assigned to the outer `delta_e`.
                } else {
                    delta_e // No cap needed, use the original `delta_e`.
                }
            } else {
                // No root satisfied the condition, so we don't apply a cap.
                delta_e // Use the original `delta_e`.
            }
        }; // The result of this block is assigned to `delta_e`.

        let omega_star = omega_trial;
        let delta_l = delta_e / omega_star;
        (e1 + delta_e, e2 - delta_e, l1 + delta_l, l2 - delta_l)

    } else {
        let (roots_array, roots_count) = cubic_y_root_cardano(x0, y0);
        
        let mut min_omega = f64::INFINITY;
        for r in roots_array.iter().take(roots_count) {
            let ell = r * ell0;
            let omega = g_times_mass.powi(2) / ell.powi(3);
            if omega > 0.0 && omega < min_omega {
                min_omega = omega;
            }
        }

        // we already know that none of our roots will have imaginary components
        let (roots_alt_array, roots_alt_count) = cubic_y_root_cardano(x0_alt, y0_alt);

        // Find the minimum omega from the second set
        let mut min_omega_alt = f64::INFINITY;
        for r in roots_alt_array.iter().take(roots_alt_count) {
            let ell = r * ell0;
            let omega = g_times_mass.powi(2) / (ell).powi(3);
            if omega > 0.0 && omega < min_omega_alt {
                min_omega_alt = omega;
            }
        }

        let omega_star = min_omega.min(min_omega_alt);

        let delta_l = delta_e / omega_star;
        (e1+delta_e, e2 - delta_e, l1+delta_l, l2-delta_l)
    }
}

/// Calculate new orb_a and ecc values for two objects that dynamically interact
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature=(smbh_mass, orb_a_give, orb_a_take, mass_give, mass_take, ecc_give, ecc_take, radius_give, radius_take, id_num_give, id_num_take, delta_energy_strong, flag_obj_types))]
pub fn encounters_new_orba_ecc(
    smbh_mass: f64,
    orb_a_give: f64,
    orb_a_take: f64,
    mass_give: f64,
    mass_take: f64,
    ecc_give: f64,
    ecc_take: f64,
    radius_give: f64,
    radius_take: f64,
    id_num_give: u32,
    id_num_take: u32,
    delta_energy_strong: f64,
    flag_obj_types: u32, // should be u64 for memory layout?

) -> (f64, f64, f64, f64, Option<u32>, Option<u32>) {
    let smbh_mass_geometric = 1.0;
    let mass_scale = smbh_mass / 1.0; // huh????
    let orb_a_give_geometric = orb_a_give * 2.0 * smbh_mass_geometric;
    let orb_a_take_geometric = orb_a_take * 2.0 * smbh_mass_geometric;
    let mass_give_geometric = mass_give / mass_scale;
    let mass_take_geometric = mass_take / mass_scale;

    let (v_relative, v_esc_sq) = match flag_obj_types {
        0 => {
            let radius_give_geometric = radius_give * 2.0 * smbh_mass_geometric;
            let radius_take_geometric = radius_take * 2.0 * smbh_mass_geometric;
            let v_relative = (smbh_mass_geometric / orb_a_give_geometric).sqrt() - (smbh_mass_geometric / orb_a_take_geometric).sqrt();
            let v_esc_sq = smbh_mass_geometric / (radius_give_geometric.max(radius_take_geometric));

            (v_relative, v_esc_sq)
        },
        1 => {
            let v_relative = (smbh_mass_geometric / orb_a_give_geometric).sqrt() - (smbh_mass_geometric / orb_a_take_geometric).sqrt();
            let v_esc_sq = 1.0;

            (v_relative, v_esc_sq)
        },
        // what do we do if flags are 2 or 3?
        _ => todo!()
    }; 

    let e_give_initial = - mass_give_geometric * smbh_mass_geometric / (2.0 * orb_a_give_geometric);
    let e_take_initial = - mass_take_geometric * smbh_mass_geometric / (2.0 * orb_a_take_geometric);
    let j_give_initial = mass_give_geometric * (smbh_mass_geometric * orb_a_give_geometric * (1.0 - ecc_give.powi(2))).sqrt();
    let j_take_initial = mass_take_geometric * (smbh_mass_geometric * orb_a_take_geometric * (1.0 - ecc_take.powi(2))).sqrt();

    let mu_geometric = mass_give_geometric * mass_take_geometric / (mass_give_geometric + mass_take_geometric);
    let delta_e = delta_energy_strong * mu_geometric * (1.0 / ((1.0 / v_relative.powi(2)) + (1.0 / v_esc_sq)));

    // unnecessary declarations
    //
    // let id_num_unbound: Option<u32> = None;
    // let id_num_flipped_rotation: Option<u32> = None;

    let (e_give_final, e_take_final, j_give_final, j_take_final) = transition_physical_as_el(e_give_initial, j_give_initial, e_take_initial, j_take_initial, delta_e, mass_give_geometric, mass_take_geometric, None, Some(smbh_mass_geometric));


    let (orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, id_num_unbound) = if e_give_initial + delta_e > 0.0 {
        let orb_a_give_final = orb_a_give;
        let ecc_give_final = ecc_give;
        let id_num_unbound = id_num_give;
        let (orb_a_take_final, ecc_take_final) = components_from_el(e_take_final / mass_take_geometric, j_take_final / mass_take_geometric, None, smbh_mass_geometric);

        (orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, Some(id_num_unbound))

    } else if e_take_initial - delta_e > 0.0 {
        let orb_a_take_final = orb_a_take;
        let ecc_take_final = ecc_take;
        let id_num_unbound = id_num_take;
        let (orb_a_give_final, ecc_give_final) = components_from_el(e_give_final / mass_give_geometric, j_give_final / mass_give_geometric, None, smbh_mass_geometric);

        (orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, Some(id_num_unbound))

    } else {
        let (orb_a_give_final, ecc_give_final) = components_from_el(e_give_final / mass_give_geometric, j_give_final / mass_give_geometric, None, smbh_mass_geometric);
        let (orb_a_take_final, ecc_take_final) = components_from_el(e_take_final / mass_take_geometric, j_take_final / mass_take_geometric, None, smbh_mass_geometric);

        (orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, None)
    };

    let ecc_give_final = if j_give_final < 0.0 {
        0.0
    } else {
        ecc_give_final
    };

    let ecc_take_final = if j_take_final < 0.0 {
        0.0
    } else {
        ecc_take_final
    };

    let id_num_flipped_rotation = if j_give_final < 0.0 {
        Some(id_num_give)
    } else if j_take_final < 0.0 {
        Some(id_num_take)
    } else {
        None
    };

    (
        orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, id_num_unbound, id_num_flipped_rotation
    )
}


// // A struct to hold the results that aren't direct array mutations.
// struct EncounterResults {
//     newly_unbound_ids: Vec<i64>,
//     newly_flipped_ids: Vec<i64>,
//     possible_touch_pairs: Vec<[i64; 2]>,
//     frac_rhill_separations: Vec<f64>,
// }
//
// #[pyfunction]
// fn process_inner_loop(
//     // Data for the single circular star
//     circ_idx: usize,
//     circ_orb_a: f64,
//     circ_mass: f64,
//     // ... other circ_ star properties
//
//     // Data for ALL eccentric stars (as read-only NumPy arrays)
//     ecc_indices: Vec<i64>,
//     ecc_orbs_a: Vec<f64>,
//     ecc_orbs_ecc: Vec<f64>,
//     // ... other ecc_ star properties
//
//     disk_star_pro_id_nums: Vec<usize>,
//     // Pre-generated random numbers for this specific outer loop `i`
//     chances_for_this_circ: Vec<f64>,
//     energies_for_this_circ: Vec<f64>,
//
//     // The main arrays we need to MUTATE
//     mut all_orbs_a: Vec<f64>,
//     mut all_orbs_ecc: Vec<f64>,
//
//
//     mut id_nums_flipped_rotation: Vec<usize>,
//     mut id_nums_unbound: Vec<usize>,
//
//     // Other parameters
//     smbh_mass: f64,
//     // ...
// ) -> EncounterResults {
//     let mut results = EncounterResults { /* ... initialize empty vecs ... */ };
//     let mut all_a = all_orbs_a.as_mut(); // Get a mutable view
//     let mut all_ecc = all_orbs_ecc.as_mut();
//
//     // The fast Rust loop
//     for (j, &ecc_idx) in ecc_indices.as_array().iter().enumerate() {
//         // ... perform all the logic from your Python inner loop ...
//         // ... using the passed-in data ...
//
//         if !id_nums_flipped_rotation.contains(disk_star_pro_id_nums[ecc_idx]) 
//             && !id_nums_flipped_rotation.contains(disk_star_pro_id_nums[&circ_idx])
//             && !id_nums_unbound.contains(disk_star_pro_id_nums[&circ_idx])
//             && !id_nums_unbound.contains(disk_star_pro_id_nums[ecc_idx]) {
//
//         }
//         // if (disk_star_pro_id_nums[ecc_idx] not in id_nums_flipped_rotation) and
//         //     (disk_star_pro_id_nums[circ_idx] not in id_nums_flipped_rotation) and
//         //     (disk_star_pro_id_nums[circ_idx] not in id_nums_unbound) and
//         //     (disk_star_pro_id_nums[ecc_idx] not in id_nums_unbound):
//         if an_encounter_happens {
//             // ... call your existing helper `encounters_new_orba_ecc` ...
//             // which could also be a pure Rust function now.
//
//             // Mutate the arrays directly
//             all_a[ecc_idx as usize] = new_orb_a_ecc;
//             all_a[circ_idx] = new_orb_a_circ;
//             all_ecc[ecc_idx as usize] = new_ecc_ecc;
//             all_ecc[circ_idx] = new_ecc_circ;
//
//             // Collect other results
//             // results.newly_unbound_ids.push(id_num_out);
//         }
//     }
//
//     results
// }
