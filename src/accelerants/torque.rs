use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

const M_SUN_KG: f64 = 1.9884099e30;  // Solar mass in kg
const G_SI: f64 = 6.67430e-11;     // Gravitational constant in m^3/(kg s^2)

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn torque_mig_timescale_helper<'py>(
    py: Python<'py>,
    smbh_mass_msun: f64,
    orbs_a_rg_arr: PyReadonlyArray1<f64>,
    masses_msun_arr: PyReadonlyArray1<f64>,
    orbs_ecc_arr: PyReadonlyArray1<f64>,
    orb_ecc_crit: f64,
    migration_torque_arr: PyReadonlyArray1<f64>,
    r_g_in_meters: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let orbs_a_rg = orbs_a_rg_arr.as_slice().unwrap();
    let masses_msun = masses_msun_arr.as_slice().unwrap();
    let orbs_ecc = orbs_ecc_arr.as_slice().unwrap();
    let migration_torque = migration_torque_arr.as_slice().unwrap();

    let smbh_mass_kg = smbh_mass_msun * M_SUN_KG;

    // First pass: identify migrating indices
    let migration_indices: Vec<usize> = orbs_ecc.iter()
        .enumerate()
        .filter_map(|(i, &ecc)| if ecc <= orb_ecc_crit { Some(i) } else { None })
        .collect();

    if migration_indices.is_empty() {
        return Ok(PyArray1::zeros(py, 0, false));
    }

    // migration_torque is already sized to len(migration_indices) from upstream,
    // so torque index j corresponds to migration_indices[j]
    debug_assert_eq!(
        migration_torque.len(),
        migration_indices.len(),
        "migration_torque length {} != migration count {}",
        migration_torque.len(),
        migration_indices.len()
    );

    let result_arr = unsafe { PyArray1::new(py, migration_indices.len(), false) };
    let result_slice = unsafe { result_arr.as_slice_mut().unwrap() };

    for (j, &idx) in migration_indices.iter().enumerate() {
        let torque_nm = migration_torque[j];

        if torque_nm == 0.0 || !torque_nm.is_finite() {
            result_slice[j] = 0.0;
            continue;
        }

        let a_m = orbs_a_rg[idx] * r_g_in_meters;
        let mass_kg = masses_msun[idx] * M_SUN_KG;

        // Omega = sqrt(G * M_smbh / a^3) in s^-1
        let omega_bh = (G_SI * smbh_mass_kg / a_m.powi(3)).sqrt();

        // t_mig = m * Omega * a^2 / (2 * Gamma_tot) in seconds
        let t_mig = mass_kg * omega_bh * a_m.powi(2) / (2.0 * torque_nm);

        result_slice[j] = if t_mig.is_finite() { t_mig } else { 0.0 };
    }

    Ok(result_arr)
}
