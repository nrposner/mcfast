use std::f64::consts::PI;

use pyo3::{prelude::*, types::PyList};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};


#[allow(clippy::collapsible_if)]
#[allow(clippy::too_many_arguments)]
pub fn circ_prograde_stars<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    solar_mass: f64,
    bin_mass_1: PyReadonlyArray1<'py, f64>,
    bin_id_nums: PyReadonlyArray1<'py, f64>,
    bin_masses: PyReadonlyArray1<'py, f64>,
    bin_sep: PyReadonlyArray1<f64>,
    bin_ecc: PyReadonlyArray1<f64>,
    bin_orb_ecc: PyReadonlyArray1<f64>,
    bin_orb_a: PyReadonlyArray1<'py, f64>,
    bin_contact_sep: PyReadonlyArray1<'py, f64>,
    bin_hill_sep: PyReadonlyArray1<'py, f64>,
    epsilon_orb_a: PyReadonlyArray1<'py, f64>,
    bin_orbits_per_timestep: PyReadonlyArray1<'py, f64>,
    ecc_orb_max: PyReadonlyArray1<'py, f64>,
    ecc_orb_min: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_locations: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_eccentricities: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_masses: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_id_nums: PyReadonlyArray1<'py, f64>,
    chances: PyReadonlyArray2<'py, f64>,
    bin_velocities: PyReadonlyArray1<'py, f64>,
    circ_velocities: PyReadonlyArray1<'py, f64>,
    bin_binding_energy: PyReadonlyArray1<'py, f64>,
    de_strong: f64,
    delta_energy_strong: f64,
    disk_radius_outer: f64,
    epsilon: f64,
) -> (Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>) {
// ) -> Bound<'py, PyList> {

    let bin_sep_slice = unsafe {bin_sep.as_slice_mut().unwrap() };
    let bin_ecc_slice = unsafe {bin_ecc.as_slice_mut().unwrap() };
    let bin_orb_ecc_slice = unsafe {bin_orb_ecc.as_slice_mut().unwrap() };

    let circ_prograde_population_locations_slice = unsafe { circ_prograde_population_locations.as_slice_mut().unwrap() };
    let circ_prograde_population_eccentricities_slice = unsafe { circ_prograde_population_eccentricities.as_slice_mut().unwrap() };

    // todo: double check these don't need to be mutable
    // buckets for appending, lists, inefficient
    let id_nums_poss_touch = PyList::empty(py);
    let frac_rhill_sep = PyList::empty(py);
    let id_nums_ionized_bin = PyList::empty(py);
    let id_nums_merged_bin = PyList::empty(py);

    // double loop, potential for a sort+sweep here?
    for i in 0..bin_mass_1.len().unwrap() {
        for j in 0..circ_prograde_population_locations.len().unwrap() {
            if !id_nums_ionized_bin.contains(bin_id_nums.get(i).unwrap()).unwrap() && !id_nums_merged_bin.contains(bin_id_nums.get(i).unwrap()).unwrap() {
                if (1.0 - *bin_orb_ecc.get(i).unwrap()) * *bin_orb_a.get(i).unwrap() < *ecc_orb_max.get(j).unwrap() && (1.0 + bin_orb_ecc.get(i).unwrap() * bin_orb_a.get(i).unwrap() > *ecc_orb_min.get(j).unwrap()) {
                    // temp_bin_mass / (3.0 * smbh_mass)
                    let bh_smbh_mass_ratio = (bin_masses.get(i).unwrap() + circ_prograde_population_masses.get(j).unwrap())/(3.0 * smbh_mass);
                    let prob_enc_per_timestep = ((1.0/PI) * (bh_smbh_mass_ratio.powf(1.0/3.0)) * bin_orbits_per_timestep.get(i).unwrap()).clamp(-100.0, 1.0);

                    // double check this syntax is right
                    let chance_of_encounter = chances.get([i, j]).unwrap();
                    if *chance_of_encounter < prob_enc_per_timestep {
                        let rel_vel_ms = (bin_velocities.get(i).unwrap() - circ_velocities.get(j).unwrap()).abs();
                        let ke_interloper = 0.5 * circ_prograde_population_masses.get(j).unwrap() * solar_mass * (rel_vel_ms.powi(2));
                        let hard = bin_binding_energy.get(i).unwrap() - ke_interloper;
                        if hard > 0.0 {
                            // these need to be mutable slices
                            bin_sep_slice[i] *= 1.0 - de_strong;
                            bin_ecc_slice[i] *= 1.0 + de_strong;
                            bin_orb_ecc_slice[i] *= 1.0 + delta_energy_strong;

                            // changing the interloper parameters
                            // this might be troublesome
                            circ_prograde_population_locations_slice[j] *= 1.0 + delta_energy_strong;
                            if circ_prograde_population_locations_slice[j] > disk_radius_outer {
                                circ_prograde_population_locations_slice[j] = disk_radius_outer - epsilon_orb_a.get(j).unwrap();
                            }
                            circ_prograde_population_eccentricities_slice[j] *= 1.0 + delta_energy_strong;
                            if bin_sep.get(i).unwrap() <= bin_contact_sep.get(i).unwrap() {
                                let _ = id_nums_merged_bin.append(*bin_id_nums.get(i).unwrap());
                            }
                        } else if hard < 0.0 {
                            bin_sep_slice[i] *= 1.0 - delta_energy_strong;
                            bin_ecc_slice[i] *= 1.0 + delta_energy_strong;
                            bin_orb_ecc_slice[i] *= 1.0 + delta_energy_strong;

                            circ_prograde_population_locations_slice[j] *= 1.0 - delta_energy_strong;
                            if circ_prograde_population_locations_slice[j] > disk_radius_outer {
                                circ_prograde_population_locations_slice[j] = disk_radius_outer - epsilon_orb_a.get(j).unwrap();
                            }
                            circ_prograde_population_eccentricities_slice[j] *= 1.0 - delta_energy_strong;

                            if bin_sep_slice[i] > *bin_hill_sep.get(i).unwrap() {
                                let _ = id_nums_ionized_bin.append(*bin_id_nums.get(i).unwrap());
                            }
                        }
                        // todo... not quite identical behavior, but prob fine??
                        // nah, make it identical
                        if *bin_ecc.get(i).unwrap() > 1.0 {
                            bin_ecc_slice[i] = 1.0 - epsilon;
                        }
                        if *bin_orb_ecc.get(i).unwrap() > 1.0 {
                            bin_orb_ecc_slice[i] = 1.0 - epsilon;
                        }
                        if *circ_prograde_population_eccentricities.get(i).unwrap() > 1.0 {
                            circ_prograde_population_eccentricities_slice[i] = 1.0 - epsilon;
                        }

                        let separation = (circ_prograde_population_locations.get(j).unwrap() - bin_orb_a.get(i).unwrap()).abs();
                        // perform a weighted average
                        // double check that this is equivalent to 
                        // center_of_mass = np.average([circ_prograde_population_locations[j], bin_orb_a[i]],
                        //                             weights=[circ_prograde_population_masses[j], bin_masses[i]])
                        let center_of_mass = ((circ_prograde_population_locations.get(j).unwrap() * circ_prograde_population_masses.get(j).unwrap()) + (bin_orb_a.get(i).unwrap() * bin_masses.get(i).unwrap())) / (circ_prograde_population_masses.get(j).unwrap() + bin_masses.get(i).unwrap());
                        let rhill_poss_encounter = center_of_mass * ((circ_prograde_population_masses.get(j).unwrap() + bin_masses.get(i).unwrap()) / (3. * smbh_mass)).powf(1.0/3.0);

                        if separation - rhill_poss_encounter < 0.0 {
                            let _ = id_nums_poss_touch.append(vec![circ_prograde_population_id_nums.get(j).unwrap(), bin_id_nums.get(i).unwrap()]);
                            let _ = frac_rhill_sep.append(separation/rhill_poss_encounter);
                        }
                    }
                }
            }
        }
    }

    // we only need to return these here, the others we've mutated in place
    (id_nums_poss_touch, frac_rhill_sep, id_nums_ionized_bin, id_nums_merged_bin)
}






