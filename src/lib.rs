use pyo3::prelude::*;

pub mod cubes;
use cubes::encounters_new_orba_ecc;

/// A Python module implemented in Rust.
#[pymodule]
fn mcfacts_helper(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encounters_new_orba_ecc, m)?)?;
    Ok(())
}
