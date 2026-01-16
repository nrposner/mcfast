use pyo3::prelude::*;

mod tools;
mod accelerants;

use accelerants::cubes::{encounters_new_orba_ecc, cubic_y_root_cardano, cubic_finite_step_root_cardano, transition_physical_as_el};
use accelerants::powerlaw::{continuous_broken_powerlaw, dual_powerlaw, dual_powerlaw_with_grid};
use accelerants::tau::{tau_ecc_dyn_helper, tau_inc_dyn_helper};
use accelerants::kick::analytical_kick_velocity_helper;
use tools::merge_tree::MergeForest;

/// A Python module implemented in Rust.
#[pymodule]
fn mcfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encounters_new_orba_ecc, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_finite_step_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(analytical_kick_velocity_helper, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_y_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(transition_physical_as_el, m)?)?;
    m.add_function(wrap_pyfunction!(continuous_broken_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw_with_grid, m)?)?;
    m.add_function(wrap_pyfunction!(tau_ecc_dyn_helper, m)?)?;
    m.add_function(wrap_pyfunction!(tau_inc_dyn_helper, m)?)?;
    m.add_class::<MergeForest>()?;
    Ok(())
}
