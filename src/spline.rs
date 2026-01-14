use std::f64;

use pyo3::{exceptions::PyValueError, prelude::*};
use splines::{Spline, Key, Interpolation};

#[pyclass]
struct PySplineConstructor { spline: Spline<f64, f64> }

#[pymethods]
impl PySplineConstructor {
    #[new]
    pub fn new(x_view: Vec<f64>, y_view: Vec<f64>) -> PyResult<Self> {
        if x_view.len() != y_view.len() {
            return Err(PyValueError::new_err("Input arrays must have the same length."));
        }
        if x_view.is_empty() {
             return Err(PyValueError::new_err("Input arrays cannot be empty."));
        }

        let keys = x_view.iter().zip(y_view.iter())
            .map(|(&x, &y)| Key::new(x, y, Interpolation::CatmullRom))
            .collect::<Vec<_>>();

        let spline = Spline::from_iter(keys);

        Ok(Self { spline })
    }

    // wtf is this doing?
    #[pyo3(name = "__call__")]
    pub fn __call__(&self, x: f64) -> f64 {
        let log_x = x.ln();

        let log_y = self.spline.sample(log_x).unwrap_or(f64::NEG_INFINITY);

        log_y.exp()
    }
}

