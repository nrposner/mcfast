use pyo3::prelude::*;

pub mod constants;
pub mod cubes;
pub mod surrogate;
pub mod luminosity;
pub mod powerlaw;
pub mod prograde;
pub mod spline;
pub mod tau;
use cubes::{encounters_new_orba_ecc, cubic_y_root_cardano, cubic_finite_step_root_cardano, transition_physical_as_el};
use powerlaw::{continuous_broken_powerlaw, dual_powerlaw, dual_powerlaw_with_grid};
use tau::{tau_p_dyn_rs, tau_inc_helper};

/// A Python module implemented in Rust.
#[pymodule]
fn mcfacts_helper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encounters_new_orba_ecc, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_finite_step_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(cubic_y_root_cardano, m)?)?;
    m.add_function(wrap_pyfunction!(transition_physical_as_el, m)?)?;
    m.add_function(wrap_pyfunction!(continuous_broken_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw, m)?)?;
    m.add_function(wrap_pyfunction!(dual_powerlaw_with_grid, m)?)?;
    m.add_function(wrap_pyfunction!(tau_p_dyn_rs, m)?)?;
    m.add_function(wrap_pyfunction!(tau_inc_helper, m)?)?;
    Ok(())
}
