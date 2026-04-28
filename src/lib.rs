use pyo3::prelude::*;

mod tools;
mod accelerants;

use accelerants::{
    baruteau::baruteau_helper,
    cubes::{encounters_new_orba_ecc_helper, cubic_y_root_cardano, cubic_finite_step_root_cardano},
    evolution::evolution_helper,
    powerlaw::{generate_r, sample_powerlaw_icdf},
    tau::{tau_ecc_dyn_helper, tau_inc_dyn_helper},
    kick::{analytical_kick_velocity_helper, merged_orb_ecc_helper},
    torque::torque_mig_timescale_helper,
    luminosity::{shock_luminosity_helper, jet_luminosity_helper},
    gw::gw_strain_helper,
    star_mass::{star_wind_mass_loss_helper, accrete_star_mass_helper},
    prograde::encounters_prograde_sweep_helper,
    units::{si_from_r_g_helper, r_g_from_units_helper, r_schwarzschild_of_m_helper},
    tde::{tde_helper, tde_helper_variant}
};
use tools::merge_tree::MergeForest;

/// A Python module implemented in Rust.
#[pymodule]
fn mcfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(baruteau_helper, m)?)?;
    m.add_function(wrap_pyfunction!(encounters_new_orba_ecc_helper, m)?)?;
    m.add_function(wrap_pyfunction!(evolution_helper, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_finite_step_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(analytical_kick_velocity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(merged_orb_ecc_helper, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_y_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(generate_r, m)?)?;
    m.add_function(wrap_pyfunction!(sample_powerlaw_icdf, m)?)?;
    m.add_function(wrap_pyfunction!(tau_ecc_dyn_helper, m)?)?;
    m.add_function(wrap_pyfunction!(tau_inc_dyn_helper, m)?)?;
    m.add_function(wrap_pyfunction!(torque_mig_timescale_helper, m)?)?;
    m.add_function(wrap_pyfunction!(jet_luminosity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(shock_luminosity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(gw_strain_helper, m)?)?;
    m.add_function(wrap_pyfunction!(star_wind_mass_loss_helper, m)?)?;
    m.add_function(wrap_pyfunction!(accrete_star_mass_helper, m)?)?;
    m.add_function(wrap_pyfunction!(encounters_prograde_sweep_helper, m)?)?;
    m.add_function(wrap_pyfunction!(r_schwarzschild_of_m_helper, m)?)?;
    m.add_function(wrap_pyfunction!(si_from_r_g_helper, m)?)?;
    m.add_function(wrap_pyfunction!(r_g_from_units_helper, m)?)?;
    m.add_function(wrap_pyfunction!(tde_helper, m)?)?;
    m.add_function(wrap_pyfunction!(tde_helper_variant, m)?)?;
    m.add_class::<MergeForest>()?;
    Ok(())
}


