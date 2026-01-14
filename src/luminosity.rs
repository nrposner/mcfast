#![allow(dead_code)]

// setup mstar runs
// dont worry abt setup scripts, gives us a bunch of mass bins?
// some of these, especially at the higher mass end, the higher the mas sof ht egalazy, the longer
// it takes to run, some of these take 3 hours to run
// just bc there's more objects in the disk
//
// barry would also be interested in seeing how dynamics, where the bottlenecks are there
// which scripts? Just mcfacts_sim
//
// make plots is the most common according to harry


use std::f64::consts::PI;

use pyo3::pyfunction;

/// A module to hold astrophysical constants in CGS units.
mod consts {
    /// Gravitational constant in cm^3 g^-1 s^-2
    pub const G_CGS: f64 = 6.67430e-8;
    /// Speed of light in cm s^-1
    pub const C_CGS: f64 = 2.99792458e10;
    /// Mass of the Sun in grams
    pub const M_SUN_CGS: f64 = 1.98847e33;
}

// placeholder fn
fn si_from_r_g(_smbh_mass: f64, r_g: f64) -> f64 {
    r_g * 1.48e5
}

// #[pyfunction]
// pub fn shock_luminosity(
//     smbh_mass: f64,
//     mass_final: f64,
//     bin_orb_a: f64,
//     disk_aspect_ratio: impl Fn(f64) -> f64,
//     disk_density: impl Fn(f64) -> f64,
//     v_kick: f64,
// ) -> f64 {
//
//     let r_hill_rg = bin_orb_a * ((mass_final / smbh_mass) / 3.0).powf(1.0/3.0); 
//     let r_hill_m = si_from_r_g(smbh_mass, r_hill_rg);
//     let r_hill_cm = r_hill_m * 100.0;
//
//     let disk_height_rg = disk_aspect_ratio(bin_orb_a) * bin_orb_a;
//     let disk_height_m = si_from_r_g(smbh_mass, disk_height_rg);
//     let disk_height_cm = disk_height_m * 100.0; 
//
//     let v_hill_cm3 = (4.0/3.0) * PI * r_hill_cm.powi(3);
//     let r_minus_h = r_hill_cm - disk_height_cm;
//     let v_subtrahend = (2.0 / 3.0) * PI * r_minus_h.powi(2) * (3.0 * r_hill_cm - r_minus_h);
//     let v_hill_gas_cm3 = (v_hill_cm3 - v_subtrahend).abs();
//
//     let disk_density_si = disk_density(bin_orb_a);
//     // todo
//     let disk_density_cgs = disk_density_si * 0.001;
//
//     let r_hill_mass_grams = disk_density_cgs * v_hill_gas_cm3;
//     let r_hill_mass_solar = r_hill_mass_grams / consts::M_SUN_CGS;
//
//     let smbh_mass_cgs = smbh_mass * consts::M_SUN_CGS;
//     let rg = (consts::G_CGS * smbh_mass_cgs) / consts::C_CGS;
//
//     let v_kick_scale = 200.0;
//
//     let energy = 1e46 * (r_hill_mass_solar * rg) * (v_kick / v_kick_scale).powi(2);
//
//     let time = 31556952.0 * ((r_hill_rg / 3.0 * rg) / (v_kick / v_kick_scale));
//     // l_shock
//     energy / time
// }

// revise

// fn jet_luminosity(
//     mass_final: f64,
//     bin_orb_a: f64,
//     disk_density: impl Fn(f64) -> f64,
//     disk_aspect_ratio: impl Fn(f64) -> f64,
//     smbh_mass: f64,
//     spin_final: f64,
//     v_kick: f64,
// ) -> f64 {
//     let disk_density_si = disk_density(bin_orb_a);
//     // todo
//     let disk_density_cgs = disk_density_si * 0.001;
//
//     // todo
//     let v_kick_scale = 200.0;
//
//     let eta = spin_final.powi(2);
//
//     let ljet =  2.5e45 * (eta / 0.1) * (mass_final / 100.0 * consts::M_SUN_CGS).powi(2) * (v_kick / v_kick_scale).powf(1.0/3.0) * (disk_density_cgs / 10e-10 * (u.g / u.cm *3));  // # Jet luminosity
//
//     ljet
// }



// one of the disk aspect ratio lambdas, from construct_disk_direct:

    // disk_surf_dens_func_log = scipy.interpolate.CubicSpline(
    //     np.log(trunc_surf_density_data[0]), np.log(trunc_surf_density_data[1]))
    // disk_surf_dens_func = lambda x, f=disk_surf_dens_func_log: np.exp(f(np.log(x)))



//     disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)
//     disk_density_cgs = disk_density_si.cgs
//
//     v_kick = v_kick * (u.km / u.s)
//     v_kick_scale = 200. * (u.km / u.s)
//
//     eta = spin_final**2
//
//     Ljet = 2.5e45 * (eta / 0.1) * (mass_final / 100 * u.M_sun)**2 * (v_kick / v_kick_scale)**-3 * (disk_density_cgs / 10e-10 * (u.g / u.cm *3))  # Jet luminosity
//     return Ljet.value




// def jet_luminosity(mass_final,
//         bin_orb_a,
//         disk_density,
//         disk_aspect_ratio,
//         smbh_mass,
//         spin_final,
//         v_kick):
//     """
//     Estimate the jet luminosity produced by Bondi-Hoyle-Lyttleton (BHL) accretion.
//
//     Based on Graham et al. (2020), the luminosity goes as:
//         L_BHL ≈ 2.5e45 erg/s * (η / 0.1) * (M / 100 M_sun)^2 * (v / 200 km/s)^-3 * (rho / 1e-9 g/cm^3)
//
//     Parameters:
//     ----------
//     mass_final : numpy.ndarray
//         mass of remnant post-merger (mass loss accounted for via Tichy & Maronetti 08)
//     bin_orb_a : numpy.ndarray
//         Orbital separation between the SMBH and the binary at the time of merger (in gravitational radii).
//     disk_density : callable
//         Function that returns the gas density at a given radius (in kg m^-3).
//     v_kick : numpy.ndarray
//         Kick velocity imparted to the remnant (in km/s).
//
//     Returns:
//     -------
//     LBHL : numpy.ndarray
//         Estimated jet (Bondi-Hoyle) luminosity (in erg/s).
//     """
//
//     disk_density_si = disk_density(bin_orb_a) * (u.kg / u.m**3)
//     disk_density_cgs = disk_density_si.cgs
//
//     v_kick = v_kick * (u.km / u.s)
//     v_kick_scale = 200. * (u.km / u.s)
//
//     eta = spin_final**2
//
//     Ljet = 2.5e45 * (eta / 0.1) * (mass_final / 100 * u.M_sun)**2 * (v_kick / v_kick_scale)**-3 * (disk_density_cgs / 10e-10 * (u.g / u.cm *3))  # Jet luminosity
//     return Ljet.value
