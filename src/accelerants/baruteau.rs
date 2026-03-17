#![allow(dead_code)]
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

use crate::accelerants::{C_SI, EARTH_TO_SOL, FloatArray1, G_SI, JUPITER_TO_SOL, M_SUN_G, M_SUN_KG, luminosity::si_from_r_g, units::*};

#[pyfunction]
pub fn baruteau_helper<'py>(
    py: Python<'py>,
    bin_mass_1: PyReadonlyArray1<f64>,
    bin_mass_2: PyReadonlyArray1<f64>,
    bin_sep: PyReadonlyArray1<f64>,
    bin_ecc: PyReadonlyArray1<f64>,
    bin_time_to_merger_gw: PyReadonlyArray1<f64>,
    bin_flag_merging: PyReadonlyArray1<f64>,
    bin_time_merged: PyReadonlyArray1<f64>,
    smbh_mass: f64,
    timestep_duration_yr: f64, 
    time_passed: f64,
) ->(FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>, FloatArray1<'py>) {

    let bin_mass_1_slice = bin_mass_1.as_slice().unwrap();
    let bin_mass_2_slice = bin_mass_2.as_slice().unwrap();
    let bin_sep_slice = bin_sep.as_slice().unwrap();
    let bin_ecc_slice = bin_ecc.as_slice().unwrap();
    let bin_time_to_merger_gw_slice = bin_time_to_merger_gw.as_slice().unwrap();
    let bin_flag_merging_slice = bin_flag_merging.as_slice().unwrap();
    let bin_time_merged_slice = bin_time_merged.as_slice().unwrap();

    // initializing return outputs from slice copies
    let out_time_to_merger_gw = PyArray1::from_slice(py, bin_time_to_merger_gw_slice);
    let out_time_to_merger_gw_slice = unsafe { out_time_to_merger_gw.as_slice_mut().unwrap() };

    let out_flag_merging = PyArray1::from_slice(py, bin_flag_merging_slice);
    let out_flag_merging_slice = unsafe { out_flag_merging.as_slice_mut().unwrap() };

    let out_sep = PyArray1::from_slice(py, bin_sep_slice);
    let out_sep_slice = unsafe { out_sep.as_slice_mut().unwrap() };

    let out_time_merged = PyArray1::from_slice(py, bin_time_merged_slice);
    let out_time_merged_slice = unsafe { out_time_merged.as_slice_mut().unwrap() };

    bin_mass_1_slice.iter()
        .zip(bin_mass_2_slice)
        .zip(bin_sep_slice)
        .zip(bin_ecc_slice)
        .enumerate()
        .for_each(|(i, (((m1, m2), sep), ecc))| { 
            if bin_flag_merging_slice[i] >= 0.0 { 
                let mass_binary = m1 + m2;

                let numerator = (1.0 - ecc.powi(2)).powf(3.5);
                let denominator = 1.0 + ((73.0/24.0) * ecc.powi(2)) + ((37.0/96.0) * ecc.powi(4));

                let ecc_factor = numerator / denominator;

                let bin_period = 0.32 * sep.powf(1.5) * (smbh_mass/1.0e8).powf(1.5) * (mass_binary/10.0).powf(-0.5);

                let scaled_num_orbit = (timestep_duration_yr / bin_period) / 1000.0;

                // equivalent to 
                //  sep_crit = (point_masses.r_schwarzschild_of_m(bin_mass_1[idx_non_mergers]) +
                //      point_masses.r_schwarzschild_of_m(bin_mass_2[idx_non_mergers]))
                // since r_schwarzschild_of_m is commutative      
                let sep_crit = r_schwarzschild_of_m_local(mass_binary);

                let sep_init = si_from_r_g(smbh_mass, *sep);

                let time_to_merger_gw: f64 = time_of_orbital_shrinkage(
                    *m1,
                    *m2,
                    sep_init,
                    sep_crit,
                ) * ecc_factor;

                out_time_to_merger_gw_slice[i] = time_to_merger_gw;

                let timestep_duration_sec = timestep_duration_yr * 31557600.0;
                let merge_mask = time_to_merger_gw <= timestep_duration_sec;

                if !merge_mask {
                    out_sep_slice[i] = *sep  * (0.5f64.powf(scaled_num_orbit));
                } else {
                    out_flag_merging_slice[i] = -2.0f64;
                    out_time_merged_slice[i] = time_passed;
                }
            } 
    });

    (out_sep, out_flag_merging, out_time_merged, out_time_to_merger_gw)
}


fn time_of_orbital_shrinkage(
    mass_1: f64, // solmass
    mass_2: f64, // solmass
    sep_initial: f64, // meters
    sep_final: f64, // meters
) -> f64 {
    // damn! we can't make this const
    let g_c = (64.0 / 5.0) * G_SI.powi(3) * C_SI.powi(-5);

    let mass_1 = mass_1 * M_SUN_KG;
    let mass_2 = mass_2 * M_SUN_KG;

    let beta = g_c * mass_1 * mass_2 * (mass_1 + mass_2);
    (sep_initial.powi(4) - sep_final.powi(4)) / 4.0 / beta
}


// gonna need to get a version of r_schwarzschild_of_m


/// Scalar version of r_schwarzschild_of_m for python-facing... or should vector version be
/// default?? 
#[pyfunction]
pub fn r_schwarzschild_of_m_helper<'py>(py: Python<'py>, mass: &Bound<'_, PyAny>) -> PyResult<FloatArray1<'py>> {

    // need to extract the unit attribute, if it exists
    // and make sure it's in solar masses

    // let solmass =
    // if let Ok(quantity) = extract_scalar_unit(mass) {
    //     match quantity.unit {
    //         Unit::Kilogram => quantity.value / KG_TO_SOL,
    //         Unit::EarthMass => quantity.value / EARTH_TO_SOL,
    //         Unit::JupiterMass => quantity.value / JUPITER_TO_SOL,
    //         Unit::SolarMass => quantity.value,
    //         _ => return Err(
    //             pyo3::exceptions::PyValueError::new_err(
    //                 format!("Unsupported unit for r_schwarzschild_of_m: {:?}", quantity.unit)
    //             )
    //         )
    //     }
    // } else 
    if let Ok(quantity) = extract_array_unit(mass) {
        let out = unsafe {PyArray1::new(py, quantity.value.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        let temp = mass.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
        let mass = temp.as_slice().unwrap();

        mass.iter().enumerate().for_each(|(i, val)| {
            let solmass = match quantity.unit {
                Unit::Gram => {
                    val / M_SUN_G
                }
                Unit::Kilogram => {
                    val / M_SUN_KG
                },
                Unit::EarthMass => {
                    val / EARTH_TO_SOL
                },
                Unit::JupiterMass => {
                    val / JUPITER_TO_SOL
                },
                Unit::SolarMass => {
                    *val
                },
                _ => panic!("Unsupported unit for r_schwarzschild_of_m: {:?}", quantity.unit)
                // _ => return Err(
                //     pyo3::exceptions::PyValueError::new_err(
                //         format!("Unsupported unit for r_schwarzschild_of_m: {:?}", quantity.unit)
                //     )
                // )

            };

            let r_sch = (2.0 * G_SI * solmass / (C_SI.powi(2))) * M_SUN_KG;

            out_slice[i] = r_sch;
        });

        Ok(out)
    } else if let Ok(masses) = mass.extract::<PyReadonlyArray1<f64>>() { 

        let out = unsafe {PyArray1::new(py, masses.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // no unit, so just assume it's in solar masses already
        masses.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            let r_sch = (2.0 * G_SI * val / (C_SI.powi(2))) * M_SUN_KG;
            out_slice[i] = r_sch;
        });

        Ok(out)
    } else {
        panic!("What the hell?")
    }


    // // using kg_to_sol here is equivalent to using .to(u.m) 
    // let r_sch = (2.0 * G_SI * solmass / (C_SI.powi(2))) * KG_TO_SOL;
    //
    // Ok(r_sch)
}


/// Assume that the value starts in solar masses
#[pyfunction]
fn r_schwarzschild_of_m_local(mass: f64) -> f64 {
    // using kg_to_sol here is equivalent to using .to(u.m) 
    (2.0 * G_SI * mass / (C_SI.powi(2))) * M_SUN_KG
}

