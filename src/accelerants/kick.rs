use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::accelerants::{FloatArray1, G_SI, M_SUN_KG, luminosity::si_from_r_g};

#[pyfunction]
pub fn analytical_kick_velocity_helper<'py>(
    py: Python<'py>,
    mass1_arr: PyReadonlyArray1<f64>,
    mass2_arr: PyReadonlyArray1<f64>,
    spin1_arr: PyReadonlyArray1<f64>,
    spin2_arr: PyReadonlyArray1<f64>,
    spin_angle1_arr: PyReadonlyArray1<f64>,
    spin_angle2_arr: PyReadonlyArray1<f64>,
    angle_arr: PyReadonlyArray1<f64>,
) -> FloatArray1<'py> {
    
    let m1_slice = mass1_arr.as_slice().unwrap();
    let m2_slice = mass2_arr.as_slice().unwrap();
    let s1_slice = spin1_arr.as_slice().unwrap();
    let s2_slice = spin2_arr.as_slice().unwrap();
    let sa1_slice = spin_angle1_arr.as_slice().unwrap();
    let sa2_slice = spin_angle2_arr.as_slice().unwrap();
    let angle_slice = angle_arr.as_slice().unwrap();

    let out_arr = unsafe{ PyArray1::new(py, m1_slice.len(), false)};
    let out_slice = unsafe{ out_arr.as_slice_mut().unwrap()};

    let xi: f64 = 145.0f64.to_radians();
    let const_a: f64 = 1.2e4;
    let const_b: f64 = -0.93;
    let const_h: f64 = 6.9e3;
    let v_11: f64 = 3678.0;
    let v_a: f64 = 2481.0;
    let v_b: f64 = 1793.0;
    let v_c: f64 = 1507.0;

    for (i, ((((((m1, m2), s1), s2), sa1), sa2), angle)) in m1_slice.iter()
        .zip(m2_slice)
        .zip(s1_slice)
        .zip(s2_slice)
        .zip(sa1_slice)
        .zip(sa2_slice)
        .zip(angle_slice)
        .enumerate() {

        // Handle the Swap (Akiba et al. Appendix A: mass_2 should be heavier)
        // We use simple variable shadowing to swap purely on the stack.
        let (loc_m1, loc_m2, loc_s1, loc_s2, loc_a1, loc_a2) = if m1 <= m2 {
            (m1, m2, s1, s2, sa1, sa2)
        } else {
            (m2, m1, s2, s1, sa2, sa1)
        };

        // Spin Components
        let (s1_sin, s1_cos) = loc_a1.sin_cos();
        let (s2_sin, s2_cos) = loc_a2.sin_cos();

        let s1_par = loc_s1 * s1_cos;
        let s1_perp = loc_s1 * s1_sin;
        let s2_par = loc_s2 * s2_cos;
        let s2_perp = loc_s2 * s2_sin;

        // Mass Ratios
        let q = loc_m1 / loc_m2;
        let q_sq = q * q;
        let one_plus_q = 1.0 + q;
        let one_plus_q_sq = one_plus_q * one_plus_q;
        
        let eta = q / one_plus_q_sq;
        let eta_sq = eta * eta;

        // Akiba Eq A5
        let s_big = (2.0 * (loc_s1 + q_sq * loc_s2)) / one_plus_q_sq;
        let s_big_sq = s_big * s_big;
        let s_big_cu = s_big_sq * s_big;

        // Akiba Eq A2 (v_m)
        let term_sqrt = (1.0 - 4.0 * eta).sqrt();
        let v_m = const_a * eta_sq * term_sqrt * (1.0 + const_b * eta);

        // Akiba Eq A3 (v_perp)
        let v_perp_mag = (const_h * eta_sq / one_plus_q) * (s2_par - q * s1_par);

        // Akiba Eq A4 (v_par)
        // Note: Python code used np.abs(spin_2_perp - q * spin_1_perp)
        let term_v = v_11 + (v_a * s_big) + (v_b * s_big_sq) + (v_c * s_big_cu);
        let spin_diff_perp = (s2_perp - q * s1_perp).abs();

        let v_par = ((16.0 * eta_sq) / one_plus_q) * term_v * spin_diff_perp * angle.cos();

        // Akiba Eq A1 (Total Kick)
        let (xi_sin, xi_cos) = xi.sin_cos();
        
        let term_1 = v_m + v_perp_mag * xi_cos;
        let term_2 = v_perp_mag * xi_sin;
        
        out_slice[i] = (term_1.powi(2) + term_2.powi(2) + v_par.powi(2)).sqrt();
    }

    out_arr
}

#[pyfunction]
pub fn merged_orb_ecc_helper<'py>(
    py: Python<'py>,
    bin_orbs_arr: PyReadonlyArray1<f64>,
    v_kick_arr: PyReadonlyArray1<f64>,
    smbh_mass: f64,
) -> FloatArray1<'py> { 

    let bin_orbs_slice = bin_orbs_arr.as_slice().unwrap();
    let v_kick_slice = v_kick_arr.as_slice().unwrap();

    let out_arr = unsafe{ PyArray1::new(py, bin_orbs_slice.len(), false)};
    let out_slice = unsafe{ out_arr.as_slice_mut().unwrap()};

    for (i, (bin_orb, v_kick)) in bin_orbs_slice.iter()
        .zip(v_kick_slice)
        .enumerate() {

        let orbs_a_units = si_from_r_g(smbh_mass, *bin_orb);

        // under the assumption that the output here is in m/s, since G is in SI 
        let v_kep = ((G_SI * (smbh_mass * M_SUN_KG) / orbs_a_units).sqrt()) / 1000.0;

        let merged_ecc = v_kick/v_kep;

        out_slice[i] = merged_ecc
    }

    out_arr
}

