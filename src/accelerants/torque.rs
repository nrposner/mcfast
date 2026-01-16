use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::constants::G;


#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn torque_mig_timescale_helper<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    orbs_a_si_arr: PyReadonlyArray1<f64>,
    masses_arr: PyReadonlyArray1<f64>,
    orbs_ecc_arr: PyReadonlyArray1<f64>,
    orb_ecc_crit: f64,
    migration_torque: PyReadonlyArray1<f64>,
    // r_g_in_meters: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {


    let orbs_ecc_slice  = orbs_ecc_arr.as_slice().unwrap();
    let orbs_a_si_slice  = orbs_a_si_arr.as_slice().unwrap();
    let migration_torque_slice  = migration_torque.as_slice().unwrap();
    let masses_slice  = masses_arr.as_slice().unwrap();

    let torque_mig_timescale = unsafe{ PyArray1::new(py, orbs_ecc_slice.len(), false)};

    let torque_mig_timescale_slice = unsafe {torque_mig_timescale.as_slice_mut().unwrap()};

    for (i, (((ecc, a), mig), mass)) in orbs_ecc_slice.iter()
        .zip(orbs_a_si_slice)
        .zip(migration_torque_slice)
        .zip(masses_slice)
        .enumerate() {
        // if *ecc <= orb_ecc_crit { Some(i) } else { None }
        if *ecc <= orb_ecc_crit {

            let omega_bh: f64 = (G * smbh_mass / ( a.powi(3) )).sqrt();

            torque_mig_timescale_slice[i] = if *mig == 0.0 {
                0.0
            } else {
                mass * omega_bh *  a.powi(2) / (2.0 * mig)
            };

        }     
    }
    Ok(torque_mig_timescale)
}
