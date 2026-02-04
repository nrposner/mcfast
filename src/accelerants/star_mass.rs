use pyo3::{exceptions::PyValueError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};


pub fn accrete_star_mass_helper(
    disk_star_pro_masses_arr: PyReadonlyArray1<f64>,
    disk_star_pro_orbs_arr: PyReadonlyArray1<f64>,
    // ???
    disk_star_luminosity_factor: f64,
    // ???
    disk_star_initial_mass_cutoff: f64,
    smbh_mass: f64,
    sound_speed_arr: PyReadonlyArray1<f64>,
    disk_density_arr: PyReadonlyArray1<f64>,
    timestep_duration_yr: f64,
    r_g_in_meters: f64,
) {

}
// def accrete_star_mass(disk_star_pro_masses,
//                       disk_star_pro_orbs_a,
//                       disk_star_luminosity_factor,
//                       disk_star_initial_mass_cutoff,
//                       smbh_mass,
//                       disk_sound_speed,
//                       disk_density,
//                       timestep_duration_yr,
//                       r_g_in_meters):
//     """Adds mass according to Fabj+2024 accretion rate
//
//     Takes initial star masses at start of timestep and adds mass according to Fabj+2024.
//
//     Parameters
//     ----------
//     disk_star_pro_masses : numpy.ndarray
//         Initial masses [M_sun] of stars in prograde orbits around SMBH with :obj:`float` type.
//     disk_star_eddington_ratio : float
//         Accretion rate of fully embedded stars [Eddington accretion rate].
//         1.0=embedded star accreting at Eddington.
//         Super-Eddington accretion rates are permitted.
//         User chosen input set by input file
//     mdisk_star_eddington_mass_growth_rate : float
//         Fractional rate of mass growth AT Eddington accretion rate per year (fixed at 2.3e-8 in mcfacts_sim) [yr^{-1}]
//     timestep_duration_yr : float
//         Length of timestep [yr]
//     r_g_in_meters: float
//         Gravitational radius of the SMBH in meters
//
//     Returns
//     -------
//     disk_star_pro_new_masses : numpy.ndarray
//         Masses [M_sun] of stars after accreting at prescribed rate for one timestep [M_sun] with :obj:`float` type
//
//     Notes
//     -----
//     Calculate Bondi radius: R_B = (2 G M_*)/(c_s **2) and Hill radius: R_Hill \\approx a(1-e)(M_*/(3(M_* + M_SMBH)))^(1/3).
//     Accretion rate is Mdot = (pi/f) * rho * c_s * min[R_B, R_Hill]**2
//     with f ~ 4 as luminosity dependent factor that accounts for the decrease of the accretion rate onto the star as it
//     approaches the Eddington luminosity (see Cantiello+2021), rho as the disk density, and c_s as the sound speed.
//     """
//
//     # Put things in SI units
//     star_masses_si = disk_star_pro_masses * u.solMass
//     disk_sound_speed_si = disk_sound_speed(disk_star_pro_orbs_a) * u.meter/u.second
//     disk_density_si = disk_density(disk_star_pro_orbs_a) * (u.kg / (u.m ** 3))
//     timestep_duration_yr_si = timestep_duration_yr * u.year
//
//     # Calculate Bondi and Hill radii
//     r_bondi = (2 * const.G.to("m^3 / kg s^2") * star_masses_si / (disk_sound_speed_si ** 2)).to("meter")
//     r_hill_rg = (disk_star_pro_orbs_a * ((disk_star_pro_masses / (3 * (disk_star_pro_masses + smbh_mass))) ** (1./3.)))
//     r_hill_m = si_from_r_g(smbh_mass, r_hill_rg, r_g_defined=r_g_in_meters)
//
//     # Determine which is smaller for each star
//     min_radius = np.minimum(r_bondi, r_hill_m)
//
//     # Calculate the mass accretion rate
//     mdot = ((np.pi / disk_star_luminosity_factor) * disk_density_si * disk_sound_speed_si * (min_radius ** 2)).to("kg/yr")
//
//     # Accrete mass onto stars
//     disk_star_pro_new_masses = ((star_masses_si + mdot * timestep_duration_yr_si).to("Msun")).value
//
//     # Stars can't accrete over disk_star_initial_mass_cutoff
//     disk_star_pro_new_masses[disk_star_pro_new_masses > disk_star_initial_mass_cutoff] = disk_star_initial_mass_cutoff
//
//     # Mass gained does not include the cutoff
//     mass_gained = ((mdot * timestep_duration_yr_si).to("Msun")).value
//
//     # Immortal stars don't enter this function as immortal because they lose a small amt of mass in star_wind_mass_loss
//     # Get how much mass is req to make them immortal again
//     immortal_mass_diff = disk_star_pro_new_masses[disk_star_pro_new_masses == disk_star_initial_mass_cutoff] - disk_star_pro_masses[disk_star_pro_new_masses == disk_star_initial_mass_cutoff]
//     # Any extra mass over the immortal cutoff is blown off the star and back into the disk
//     immortal_mass_lost = mass_gained[disk_star_pro_new_masses == disk_star_initial_mass_cutoff] - immortal_mass_diff
//
//     assert np.all(disk_star_pro_new_masses > 0), \
//         "disk_star_pro_new_masses has values <= 0"
//
//     return disk_star_pro_new_masses, mass_gained.sum(), immortal_mass_lost.sum()
