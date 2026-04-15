# McFAST v0.1.7

Utilities and accelerated functions designed for use by the McFACTS team at CUNY.

Functions currently fully tested for integration:
- tau_inc_dyn_helper
- tau_ecc_dyn_helper
- gw_strain_helper
- analytical_kick_velocity_helper
- merged_orb_ecc_helper
- torque_mig_timescale_helper
- generate_r
- shock_luminosity_helper
- jet_luminosity_helper
- encounters_prograde_sweep_helper
- si_from_r_g_helper
- r_g_from_units_helper
- r_schwarzschild_of_m_helper

Functions currently in testing for integration:
star_wind_mass_loss_helper
accrete_star_mass_helper

## Example: Tau Incline/Eccentricity

In order to accelerate the tau_inc_dyn and tau_ecc_dyn functions in `mcfacts/physics/disk_capture.py`, we provide `tau_inc_dyn_helper` and `tau_ecc_dyn_helper` functions that perform the vectorized array calculations in an optimized Rust function. The example implementation for `tau_inc_dyn_optimized` follows: it has the same call signature, takes the same inputs, and provides the same output as `tau_inc_dyn`, but is 2-3x faster.

This function and its `ecc` variant can be further optimized by speeding up or eliminating the calls to si_from_r_g, which now make up the bulk of the runtime in each function (approximately 2/3rds).

In particular, the `tau_semi_lat` functionality in `tau_ecc_dyn` is sped up at least 120x, with tau_semi_lat initially being timed at 6 seconds, while the `tau_ecc_dyn_helper` functionality that subsumes it is timed at 0.05 seconds.

```python
def tau_inc_dyn_optimized(smbh_mass, disk_bh_retro_orbs_a, disk_bh_retro_masses, omega, ecc, inc, disk_surf_density_func, r_g_in_meters):
    """Computes inclination damping timescale from actual variables; used only for scaling.
    Uses Rust-accelerated helper functions for the calculation: compare to tau_inc_dyn
    """
    # throw most things into SI units (that's right, ENGINEER UNITS!)
    #    or more locally convenient variable names
    SI_smbh_mass = smbh_mass * u.Msun.to("kg")  # kg
    SI_semi_maj_axis = si_from_r_g(smbh_mass, disk_bh_retro_orbs_a, r_g_defined=r_g_in_meters).to("m").value
    SI_orbiter_mass = disk_bh_retro_masses * u.Msun.to("kg")  # kg
    cos_omega = np.cos(omega)

    disk_surf_res = disk_surf_density_func(disk_bh_retro_orbs_a)

    tau_i_dyn = tau_inc_dyn_helper(SI_smbh_mass, SI_orbiter_mass, ecc, inc, cos_omega, disk_surf_res, SI_semi_maj_axis)

    assert np.isfinite(tau_i_dyn).all(), \
        "Finite check failure: tau_i_dyn"

    return tau_i_dyn
```

## Merge Tree
For M. McCarthy's black hole merge tracking feature.

```python
import mcfast
from mcfast import MergeForest

# creates a forest of all black holes
forest = MergeForest("./data/", "galaxy_state_*")

bh_uuid: str = "2b422064-0a84-4687-a542-395dcb61cd4f"

# get the immediate child of a given UUID, if the UUID is valid
descendant_uuid = forest.get_descendant(bh_uuid)

# get the immediate parents of a given UUID, if the UUID is valid
(parent1_uuid, parent2_uuid) = forest.get_parents(bh_uuid)

# get the full ancestry list of a given UUID (in DFS order), 
# assuming the UUID has ancestors
ancestor_list = forest.get_ancestors(bh_uuid)


# get the full list of black holes between the UUID and 
# the final product (root)
descent_path = forest.get_lineage_to_root(bh_uuid)

# get the generation of a given UUID, where an initialized BH is generation 0
# note: implementation doesn't fully line up with existing
# reference implementation, under review
generation = forest.get_generation(bh_uuid)

# get all 'root' black holes (products remaining at the final tick)
roots = forest.roots()

# get all 'leaf' black holes (initialized BHs with no ancestors)
leaves = forest.leaves()

# get all 'singleton' black holes (neither ancestors nor children)
singletons = forest.singletons()

if forest.contains(bh_uuid):
    print("It exists!")
else:
    print("It doesn't exist!")

# get total number of nodes in the 'forest'
total = len(forest)

# serialize and deserialize with pickle
import pickle
 
# Save now
with open("forest.pkl", "wb") as f:
    pickle.dump(forest, f)

# Load later
with open("forest.pkl", "rb") as f:
    restored = pickle.load(f)

# sample: 330-node forest serializes down to 16kb
```

