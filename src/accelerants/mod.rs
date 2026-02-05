pub mod cubes;
pub mod surrogate;
pub mod kick;
pub mod luminosity;
pub mod powerlaw;
pub mod prograde;
pub mod spline;
pub mod tau;
pub mod torque;
pub mod star_mass;
pub mod gw;

use pyo3::prelude::*;
use numpy::PyArray1;
type FloatArray1<'py> = Bound<'py, PyArray1<f64>>;

/// Gravitational constant in cm^3 g^-1 s^-2
pub const G_CGS: f64 = 6.67430e-8;
/// Speed of light in cm s^-1
pub const C_CGS: f64 = 2.99792458e10;
/// Mass of the Sun in grams
pub const _M_SUN_CGS: f64 = 1.98847e33;

pub const M_SUN_KG: f64 = 1.9884099e30;  // Solar mass in kg
pub const C_SI: f64 = 299792460.0;     // Speed of light in m/s
pub const G_SI: f64 = 6.67430e-11;     // Gravitational constant in m^3/(kg s^2)
pub const MPC_SI: f64 = 3.08568e22; // number of meters in a megaparsec
pub const L_SUN_W: f64 = 3.828e26;       // watts
pub const YR_S: f64 = 3.15576e7;         // seconds per Julian year
pub const R_SUN_M: f64 = 6.957e8;
