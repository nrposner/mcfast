
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    
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
        // --- LOGIC START ---

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

    Ok(out_arr)

}

//     # As in Akiba et al 2024 Appendix A, mass_2 should be the more massive BH in the binary.
//     mask = mass_1 <= mass_2
//
//     m_1_new = np.where(mask, mass_1, mass_2) * u.solMass
//     m_2_new = np.where(mask, mass_2, mass_1)* u.solMass
//     spin_1_new = np.where(mask, spin_1, spin_2)
//     spin_2_new = np.where(mask, spin_2, spin_1)
//     spin_angle_1_new = np.where(mask, spin_angle_1, spin_angle_2)
//     spin_angle_2_new = np.where(mask, spin_angle_2, spin_angle_1)
//
//     # "perp" and "par" refer to components perpendicular and parallel to the orbital angular momentum axis, respectively.
//     # Orbital angular momentum axis of binary is aligned with the disk angualr momentum.
//     # Find the perp and par components of spin:
//     spin_1_par = spin_1_new * np.cos(spin_angle_1_new)
//     spin_1_perp = spin_1_new * np.sin(spin_angle_1_new)
//     spin_2_par = spin_2_new * np.cos(spin_angle_2_new)
//     spin_2_perp = spin_2_new * np.sin(spin_angle_2_new)
//
//     # Find the mass ratio q and asymmetric mass ratio eta
//     # as defined in Akiba et al. 2024 Appendix A:
//     q = m_1_new / m_2_new
//     eta = q / (1 + q)**2
//
//     # Use Akiba et al. 2024 eqn A5:
//     S = (2 * (spin_1_new + q**2 * spin_2_new)) / (1 + q)**2
//
//     # As defined in Akiba et al. 2024 Appendix A:
//     xi = np.radians(145)
//     A = 1.2e4 * u.km / u.s
//     B = -0.93
//     H = 6.9e3 * u.km / u.s
//     V_11, V_A, V_B, V_C = 3678 * u.km / u.s, 2481 * u.km / u.s, 1793* u.km / u.s, 1507 * u.km / u.s
//     angle = rng.uniform(0.0, 2*np.pi, size=len(mass_1))
//
//     # Use Akiba et al. 2024 eqn A2:
//     v_m = A * eta**2 * np.sqrt(1 - 4 * eta) * (1 + B * eta)
//
//     # Use Akiba et al. 2024 eqn A3:
//     v_perp = (H * eta**2 / (1 + q)) * (spin_2_par - q * spin_1_par)
//
//     # Use Akiba et al. 2024 eqn A4:
//     v_par = ((16 * eta**2) / (1 + q)) * (V_11 + (V_A * S) + (V_B * S**2) + (V_C * S**3)) * \
//             np.abs(spin_2_perp - q * spin_1_perp) * np.cos(angle)
//
//     # Use Akiba et al. 2024 eqn A1:
//     v_kick = np.sqrt((v_m + v_perp * np.cos(xi))**2 +
//                      (v_perp * np.sin(xi))**2 +
//                      v_par**2)
//     v_kick = np.array(v_kick.value)
//     assert np.all(v_kick > 0), \
//         "v_kick has values <= 0"
//     assert np.isfinite(v_kick).all(), \
//         "Finite check failure: v_kick"
//     return v_kick
