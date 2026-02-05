use pyo3::prelude::*;

mod tools;
mod accelerants;

use accelerants::{
    cubes::{encounters_new_orba_ecc, cubic_y_root_cardano, cubic_finite_step_root_cardano, transition_physical_as_el},
    powerlaw::{continuous_broken_powerlaw, dual_powerlaw, dual_powerlaw_with_grid, generate_r},
    tau::{tau_ecc_dyn_helper, tau_inc_dyn_helper},
    kick::{analytical_kick_velocity_helper, merged_orb_ecc_helper},
    torque::torque_mig_timescale_helper,
    luminosity::{shock_luminosity_helper, jet_luminosity_helper},
    gw::gw_strain_helper,
    star_mass::{star_wind_mass_loss_helper, accrete_star_mass_helper},
};
use tools::merge_tree::MergeForest;

/// A Python module implemented in Rust.
#[pymodule]
fn mcfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encounters_new_orba_ecc, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_finite_step_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(analytical_kick_velocity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(merged_orb_ecc_helper, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_y_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(transition_physical_as_el, m)?)?;
    m.add_function(wrap_pyfunction!(continuous_broken_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw_with_grid, m)?)?;
    m.add_function(wrap_pyfunction!(generate_r, m)?)?;
    m.add_function(wrap_pyfunction!(tau_ecc_dyn_helper, m)?)?;
    m.add_function(wrap_pyfunction!(tau_inc_dyn_helper, m)?)?;
    m.add_function(wrap_pyfunction!(torque_mig_timescale_helper, m)?)?;
    m.add_function(wrap_pyfunction!(jet_luminosity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(shock_luminosity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(gw_strain_helper, m)?)?;
    m.add_function(wrap_pyfunction!(star_wind_mass_loss_helper, m)?)?;
    m.add_function(wrap_pyfunction!(accrete_star_mass_helper, m)?)?;
    m.add_class::<MergeForest>()?;
    Ok(())
}
