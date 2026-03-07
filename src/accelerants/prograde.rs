use std::{collections::BTreeSet, f64::consts::PI};

use pyo3::{prelude::*, types::PyList};
use numpy::{PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array1, Array2, Axis};
use std::cmp::Ordering;

use crate::accelerants::{C_SI, FloatArray1, G_SI};


// A particle with its original index preserved.
// No more parallel vecs — everything travels together.
#[derive(Clone)]
struct Particle {
    orig_idx: usize,
    orb_a:    f64,
    orb_ecc:  f64,
    mass:     f64,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum EventKind {
    Start, // -1: open an eccentric interval
    Point, //  0: a circular particle lives here
    End,   // +1: close an eccentric interval
}

struct Event {
    radius:   f64,
    kind:     EventKind,
    rel_idx:  usize, // index into circ[] or ecc[] depending on kind
}

impl PartialEq  for Event { fn eq(&self, o: &Self) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq         for Event {}
impl PartialOrd for Event { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by radius first, then by EventKind (Start < Point < End)
        // so that boundary-touching intervals are opened before a point is
        // processed, and closed after — matching the Python behaviour.
        self.radius
            .partial_cmp(&other.radius)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.kind.cmp(&other.kind))
    }
}

// helper function for circular_singles_encounters_prograde
#[pyfunction]
pub fn encounters_prograde_sweep_helper<'py>(
    py:                      Python<'py>,
    smbh_mass:               f64,
    disk_bh_pro_orbs_a:      PyReadonlyArray1<f64>,
    disk_bh_pro_masses:      PyReadonlyArray1<f64>,
    disk_bh_pro_orbs_ecc:    PyReadonlyArray1<f64>,
    timestep_duration_yr:    f64,
    disk_bh_pro_orb_ecc_crit: f64,
    delta_energy_strong:     f64,
    disk_radius_outer:       f64,
    // eps_denom and chance_of_enc are pre-generated on the Python side
    // and passed in so that random-state management stays in Python.
    eps_denom:               PyReadonlyArray2<f64>, // shape (N_circ, N_ecc)
    chance_of_enc:           PyReadonlyArray2<f64>, // shape (N_circ, N_ecc)
// ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {

    // ── 1. Build owned, mutable output arrays ────────────────────────────────
    // We copy once here; every mutation below operates on these Vecs.
    let mut out_orbs_a:   Vec<f64> = disk_bh_pro_orbs_a.as_slice()?.to_vec();
    let mut out_orbs_ecc: Vec<f64> = disk_bh_pro_orbs_ecc.as_slice()?.to_vec();
    let masses:           &[f64]   = disk_bh_pro_masses.as_slice()?;

    // ── 2. Partition into circular / eccentric populations ───────────────────
    // Each Particle carries its original index so we never need a separate
    // "relative → absolute" lookup table.
    let (circ, ecc): (Vec<Particle>, Vec<Particle>) = out_orbs_ecc
        .iter()
        .enumerate()
        .map(|(i, &ecc)| Particle {
            orig_idx: i,
            orb_a:    out_orbs_a[i],
            orb_ecc:  ecc,
            mass:     masses[i],
        })
        .partition(|p| p.orb_ecc <= disk_bh_pro_orb_ecc_crit);

    let circ_len = circ.len();
    let ecc_len  = ecc.len();

    if circ_len == 0 || ecc_len == 0 {
        // Nothing to do — return the (unchanged) input arrays as new Python objects.
        let a   = Array1::from(out_orbs_a).into_pyarray(py);
        let ecc = Array1::from(out_orbs_ecc).into_pyarray(py);
        return Ok((a, ecc));
    }

    // ── 3. Pre-compute per-circ orbital timescale ratio ──────────────────────
    let n_orbs_per_timestep: Vec<f64> = circ.iter().map(|p| {
        let t_orb = PI * p.orb_a.powf(1.5)
                  * (2.0e30 * smbh_mass * G_SI)
                  / (C_SI.powf(3.0) * 3.15e7);
        timestep_duration_yr / t_orb
    }).collect();

    // ── 4. Epsilon: Hill-radius × random — shape (N_circ, N_ecc) ─────────────
    // hill_radii[i] = disk_radius_outer * (m_circ / (3*(m_circ + M_smbh)))^(1/3)
    let hill_radii: Array1<f64> = Array1::from_iter(
        circ.iter().map(|p| {
            disk_radius_outer * (p.mass / (3.0 * (p.mass + smbh_mass))).powf(1.0 / 3.0)
        })
    );
    // Broadcast (N_circ, 1) * (N_circ, N_ecc)  →  (N_circ, N_ecc)
    let epsilon: Array2<f64> =
        &hill_radii.insert_axis(Axis(1)) * &eps_denom.as_array();

    let chance = chance_of_enc.as_array();

    // ── 5. Build the sweep-line event list ───────────────────────────────────
    let mut events: Vec<Event> = Vec::with_capacity(circ_len + 2 * ecc_len);

    for (rel_idx, p) in circ.iter().enumerate() {
        events.push(Event { radius: p.orb_a, kind: EventKind::Point, rel_idx });
    }
    for (rel_idx, p) in ecc.iter().enumerate() {
        let min_r = p.orb_a * (1.0 - p.orb_ecc);
        let max_r = p.orb_a * (1.0 + p.orb_ecc);
        events.push(Event { radius: min_r, kind: EventKind::Start, rel_idx });
        events.push(Event { radius: max_r, kind: EventKind::End,   rel_idx });
    }

    events.sort(); // stable sort not required; EventKind ordering handles ties

    // ── 6. Sweep ─────────────────────────────────────────────────────────────
    // BTreeSet gives us sorted iteration for free, matching Python's
    //   `sorted(list(active_ecc_indices))`.
    let mut active: BTreeSet<usize> = BTreeSet::new();

    // We need a snapshot of active indices before we can break out of the
    // inner loop, so collect into a Vec first.
    for event in &events {
        match event.kind {
            EventKind::Start => { active.insert(event.rel_idx); }
            EventKind::End   => { active.remove(&event.rel_idx); }
            EventKind::Point => {
                if active.is_empty() { continue; }

                let circ_rel_idx = event.rel_idx;
                let circ_orig    = circ[circ_rel_idx].orig_idx;

                // Snapshot so we can break cleanly without borrow issues.
                let sorted_interlopers: Vec<usize> = active.iter().copied().collect();

                for ecc_rel_idx in sorted_interlopers {
                    let ecc_orig = ecc[ecc_rel_idx].orig_idx;

                    let temp_bin_mass     = masses[circ_orig] + masses[ecc_orig];
                    let bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass);
                    let mass_ratio_factor  = bh_smbh_mass_ratio.powf(1.0 / 3.0);
                    let prob_orbit_overlap = mass_ratio_factor / PI;
                    let prob_enc = (prob_orbit_overlap * n_orbs_per_timestep[circ_rel_idx]).min(1.0);

                    if chance[[circ_rel_idx, ecc_rel_idx]] < prob_enc {
                        // Kick the circular particle into eccentricity
                        out_orbs_ecc[circ_orig] = delta_energy_strong;
                        out_orbs_a[circ_orig]  *= 1.0 + delta_energy_strong;
                        if out_orbs_a[circ_orig] >= disk_radius_outer {
                            out_orbs_a[circ_orig] =
                                disk_radius_outer - epsilon[[circ_rel_idx, ecc_rel_idx]];
                        }

                        // Damp the eccentric interloper
                        out_orbs_ecc[ecc_orig] *= 1.0 - delta_energy_strong;
                        out_orbs_a[ecc_orig]   *= 1.0 - delta_energy_strong;

                        break; // circ particle is kicked; no further encounters this step
                    }
                }
            }
        }
    }

    // ── 7. Return mutated arrays to Python ───────────────────────────────────
    let out_a   = Array1::from(out_orbs_a).into_pyarray(py);
    let out_ecc = Array1::from(out_orbs_ecc).into_pyarray(py);
    Ok((out_a, out_ecc))
}

// pub fn encounters_prograde_sweep_helper<'py>(
//     py: Python<'py>,
//     smbh_mass: f64,
//     disk_bh_pro_orbs_a: PyReadonlyArray1<f64>,
//     disk_bh_pro_masses: PyReadonlyArray1<f64>,
//     disk_bh_pro_orbs_ecc: PyReadonlyArray1<f64>,
//     timestep_duration_yr: f64,
//     disk_bh_pro_orb_ecc_crit: f64,
//     delta_energy_strong: f64,
//     disk_radius_outer: f64,
//     eps_denom: PyReadonlyArray2<f64>,
//     chance_of_enc: PyReadonlyArray2<f64>,
// ) -> PyResult<(FloatArray1<'py>, FloatArray1<'py>)> {
//
//     // let orbs_ecc = disk_bh_pro_orbs_ecc.as_slice().unwrap();
//     let mut disk_bh_pro_masses_slice = disk_bh_pro_masses.as_slice().unwrap();
//     let mut disk_bh_pro_orbs_a_slice = disk_bh_pro_orbs_a.as_slice().unwrap();
//     let mut disk_bh_pro_orbs_ecc_slice = disk_bh_pro_orbs_ecc.as_slice().unwrap();
//
//
//     // let (circ_prograde_population_indices, ecc_prograde_population_indices): (Vec<f64>, Vec<f64>) = disk_bh_pro_orbs_ecc_slice.iter().partition(|i| *i <= &disk_bh_pro_orb_ecc_crit);
//
//     // push manually??
//
//     let mut circ_prograde_population_indices = vec![];
//     let mut ecc_prograde_population_indices = vec![];
//
//     let mut circ_masses_vec = vec![];
//     let mut ecc_masses_vec = vec![];
//
//     let mut circ_orbs_a_vec = vec![];
//     let mut ecc_orbs_a_vec = vec![];
//
//     let mut circ_orbs_ecc_vec = vec![];
//     let mut ecc_orbs_ecc_vec = vec![];
//
//     disk_bh_pro_orbs_ecc_slice.iter()
//         .zip(disk_bh_pro_masses_slice)
//         .zip(disk_bh_pro_orbs_a_slice).enumerate().for_each(|(i, ((ecc, mass), orb))| {
//         if ecc <= &disk_bh_pro_orb_ecc_crit {
//             circ_prograde_population_indices.push(i);
//             circ_masses_vec.push(*mass);
//             circ_orbs_a_vec.push(*orb);
//             circ_orbs_ecc_vec.push(*ecc)
//         } else {
//             ecc_prograde_population_indices.push(i);
//             ecc_masses_vec.push(*mass);
//             ecc_orbs_a_vec.push(*orb);
//             ecc_orbs_ecc_vec.push(*ecc)
//         }
//     });
//
//     let circ_masses: Array1<f64> = Array1::from_vec( circ_masses_vec);
//     let circ_orbs: Array1<f64> = Array1::from_vec( ecc_masses_vec);
//
//     let circ_orbs_a: Array1<f64> = Array1::from_vec( circ_orbs_a_vec );
//     let ecc_orbs_a: Array1<f64> = Array1::from_vec( ecc_orbs_a_vec );
//
//     let circ_orbs_ecc: Array1<f64> = Array1::from_vec( circ_orbs_ecc_vec );
//     let ecc_orbs_ecc: Array1<f64> = Array1::from_vec( ecc_orbs_ecc_vec );
//
//
//     let circ_len = circ_prograde_population_indices.len();
//     let ecc_len = ecc_prograde_population_indices.len();
//
//     if circ_len == 0 || ecc_len == 0 {
//         return Ok((disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc))
//     }
//
//     // Step 1: compute Hill radii — shape (N_circ,)
//     let hill_radii: Array1<f64> = circ_masses.mapv(|m| {
//         disk_radius_outer * (m / (3.0 * (m + smbh_mass))).powf(1.0 / 3.0)
//     });
//
//     // Step 2: promote to column vector — shape (N_circ, 1)
//     // mirrors Python's [:, None]
//     let hill_radii_2d = hill_radii.insert_axis(Axis(1));
//
//     // Step 3: get ArrayView2 from the numpy array
//     let rng_array = eps_denom.as_array(); // ArrayView2<f64>, shape (N_circ, N_ecc)
//
//     // Step 4: broadcast multiply — ndarray handles (N_circ, 1) * (N_circ, N_ecc)
//     // automatically, mirroring numpy broadcasting
//     let epsilon: Array2<f64> = &hill_radii_2d * &rng_array;
//
//
//     let n_circ_orbs_per_timescale: Vec<f64> = circ_orbs_a.iter().map(|orbs| {
//         timestep_duration_yr / (PI * orbs.powf(1.5) * (2.0e30 * smbh_mass * G_SI) / (C_SI.powf(3.0) * 3.15e7)) 
//     }).collect();
//
//
//     // radius, type, then rel_idx
//     #[derive(PartialEq, Eq, PartialOrd)]
//     enum EventOrder {Start, Point, End}
//     #[derive(PartialEq, PartialOrd)]
//     struct Event {
//         radius: f64,
//         order: EventOrder,
//         rel_idx: usize,
//     }
//
//     impl Eq for Event {
//
//     }
//     impl Ord for Event {
//         fn cmp(&self, other: &Self) -> Ordering {
//             if self.radius < other.radius {
//                 Ordering::Less
//             } else if self.radius > other.radius {
//                 Ordering::Greater
//             } else {
//                 if self.order < other.order {
//                     Ordering::Less
//                 } else if self.order > other.order {
//                     Ordering::Greater
//                 } else {
//                     Ordering::Equal
//                 }
//             }
//         }
//     }
//
//     // fuse these together somehow??
//     // or with the below?
//     let ecc_orbs_min: Vec<f64> = ecc_orbs_a.iter().zip(ecc_orbs_ecc.iter()).map(|(orb, ecc)| {
//         orb * (1.0 - ecc)
//     }).collect();
//
//     let ecc_orbs_max: Vec<f64> = ecc_orbs_a.iter().zip(ecc_orbs_ecc.iter()).map(|(orb, ecc)| {
//         orb * (1.0 + ecc)
//     }).collect();
//
//
//     // start with the starts
//     let mut events: Vec<Event> = ecc_orbs_min.iter().enumerate().map(|(rel_idx, radius)| {
//         Event {radius: *radius, order: EventOrder::Start, rel_idx}
//         // (*radius, -1, i)
//         // EventType::Start { radius: *radius, rel_idx: i }
//     }).collect();
//
//     let points: Vec<Event> = circ_orbs_a.iter().enumerate().map(|(rel_idx, radius)| {
//         Event {radius: *radius, order: EventOrder::Point, rel_idx}
//         // (*radius, 0, i)
//         // EventType::Point {radius: *radius, rel_idx: i}
//     }).collect();
//
//     let ends: Vec<Event> = ecc_orbs_max.iter().enumerate().map(|(rel_idx, radius)| {
//         Event {radius: *radius, order: EventOrder::End, rel_idx}
//         // (*radius, 1, i)
//         // EventType::End { radius: *radius, rel_idx: i }
//     }).collect();
//
//     // join these vecs
//     events.extend(points);
//     events.extend(ends);
//
//     // now sort them
//     // use sort rather than sort_unstable? because we want to preserve order of equal elements?? or
//     // can we rely on our explicit typing for second??
//     events.sort();
//
//     let mut active_ecc_indices: BTreeSet<usize> = BTreeSet::new();
//
//     for event in events {
//         match event.order {
//             EventOrder::Start => {
//                 active_ecc_indices.insert(event.rel_idx);
//             }
//             EventOrder::End => {
//                 active_ecc_indices.remove(&event.rel_idx);
//             }
//             EventOrder::Point => {
//                 if active_ecc_indices.is_empty() {
//                     continue
//                 }
//                 let circ_rel_idx = event.rel_idx;
//
//                 let circ_idx = circ_prograde_population_indices[circ_rel_idx];
//
//                 // btreeset is already sorted??
//                 // let sorted_interlopers = active_ecc_indices.sort()
//
//                 for ecc_rel_idx in active_ecc_indices {
//                     let ecc_idx = ecc_prograde_population_indices[ecc_rel_idx];
//
//                     let temp_bin_mass = disk_bh_pro_masses_slice[circ_idx] + disk_bh_pro_masses_slice[ecc_idx];
//                     let bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass);
//                     let mass_ratio_factor = (bh_smbh_mass_ratio).powf(1.0/3.0);
//                     let prob_orbit_overlap = (1.0 / PI) * mass_ratio_factor;
//                     let prob_enc_per_timestep = (prob_orbit_overlap * n_circ_orbs_per_timescale[circ_rel_idx]).min(1.0);
//
//                     if chance_of_enc[[circ_rel_idx, ecc_rel_idx]] < prob_enc_per_timestep {
//                         disk_bh_pro_orbs_ecc_slice[circ_idx] = delta_energy_strong * 1.0001;
//                         disk_bh_pro_orbs_a_slice[circ_idx] *= (1.0 + delta_energy_strong);
//                         if disk_bh_pro_orbs_a_slice[circ_idx] >= disk_radius_outer {
//                             disk_bh_pro_orbs_a_slice = disk_radius_outer - epsilon[circ_rel_idx][ecc_rel_idx];
//                         }
//
//                         disk_bh_pro_orbs_ecc_slice[ecc_idx] *= 1.0 - delta_energy_strong;
//                         disk_bh_pro_orbs_a_slice[ecc_idx] *= 1.0 - delta_energy_strong;
//                         break
//                     }
//                 }
//             } 
//         } 
//     }
//     // have to turn these 1d arrays into bound py arrays to return to python
//     Ok((disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc))
// }

//     circ_len = len(circ_prograde_population_indices)
//     ecc_len = len(ecc_prograde_population_indices)
//     if (circ_len == 0) or (ecc_len == 0):
//         return disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc
//
//     # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
//     epsilon = (disk_radius_outer * ((disk_bh_pro_masses[circ_prograde_population_indices] /
//                (3 * (disk_bh_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)))[:, None] * \
//               rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
//
//     # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
//     #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
//     orbital_timescales_circ_pops = np.pi*((disk_bh_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*const.G.value)/(const.c.value**(3.0)*3.15e7)
//     N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
//     ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
//     ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
//     # Generate all possible needed random numbers ahead of time
//     chance_of_enc = rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
//
//     if (circ_len/(circ_len + ecc_len)) * (ecc_len/(circ_len + ecc_len)) * 100 > 50: # an ad-hoc check to see whether the double loop or sweep will be faster
//         # if True engage the sweep algorithm
//
//         # create the events array
//         # define types to ensure correct sorting at boundary conditions:
//         # START events are processed first, then POINTs, then ENDs
//         START, POINT, END = -1, 0, 1
//
//         # C = circ_prograde_population_indices.size
//         # ecc_len = ecc_prograde_population_indices.size
//
//         # create a single, flat, contiguous array for all events
//         events = np.empty(circ_len + 2 * ecc_len, dtype=[('radius', 'f8'), ('type', 'i4'), ('rel_idx', 'u4')])
//
//         # add POINT events for each circular object
//         events[:circ_len] = np.array(list(zip(disk_bh_pro_orbs_a[circ_prograde_population_indices], [POINT] * circ_len, np.arange(circ_len))), dtype=events.dtype)
//
//         # Add START and ecc_lenND events for each eccentric object's interval
//         ecc_orb_min = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0 - disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
//         ecc_orb_max = disk_bh_pro_orbs_a[ecc_prograde_population_indices] * (1.0 + disk_bh_pro_orbs_ecc[ecc_prograde_population_indices])
//         events[circ_len:circ_len+ecc_len] = np.array(list(zip(ecc_orb_min, [START] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)
//         events[circ_len+ecc_len:] = np.array(list(zip(ecc_orb_max, [END] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)
//
//         # sort the events by radius
//         # uses numpy sort, very performant
//         events.sort(order=['radius', 'type'])
//
//         # sweep and process
//         active_ecc_indices = set()
//         for radius, type, rel_idx in events:
//             if type == START:
//                 active_ecc_indices.add(rel_idx)
//             elif type == END:
//                 active_ecc_indices.discard(rel_idx) # Use discard for safety
//             elif type == POINT:
//                 # when we hit a POINT event, the `active_ecc_indices` set contains
//                 # ALL eccentric particles whose intervals contain this point
//                 if not active_ecc_indices:
//                     continue
//
//                 circ_rel_idx = rel_idx
//                 circ_idx = circ_prograde_population_indices[circ_rel_idx]
//
//                 # sort the indices to ensure deterministic processing order
//                 sorted_interlopers = sorted(list(active_ecc_indices))
//
//                 # if we remove this sort and instead just iterate directly
//                 # over active_ecc_indices, we unlock another 2x+ improvement in performance
//                 # but at the cost of genuinely massively deviating values
//
//                 for ecc_rel_idx in sorted_interlopers:
//                     ecc_idx = ecc_prograde_population_indices[ecc_rel_idx]
//
//                     temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
//                     bh_smbh_mass_ratio = temp_bin_mass / (3.0 * smbh_mass)
//                     mass_ratio_factor = (bh_smbh_mass_ratio)**(1. / 3.)
//                     prob_orbit_overlap = (1. / np.pi) * mass_ratio_factor
//                     prob_enc_per_timestep = min(prob_orbit_overlap * N_circ_orbs_per_timestep[circ_rel_idx], 1.0)
//
//                     if chance_of_enc[circ_rel_idx, ecc_rel_idx] < prob_enc_per_timestep:
//                         # apply state change, using the fixed logic
//                         disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong * 1.0001
//                         disk_bh_pro_orbs_a[circ_idx] *= (1.0 + delta_energy_strong)
//                         if (disk_bh_pro_orbs_a[circ_idx] >= disk_radius_outer):
//
//                             disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[circ_rel_idx][ecc_rel_idx]
//
//                         disk_bh_pro_orbs_ecc[ecc_idx] *= (1.0 - delta_energy_strong)
//                         disk_bh_pro_orbs_a[ecc_idx] *= (1.0 - delta_energy_strong)
//                         # Once the circular BH is kicked, break from this inner loop
//                         # as it can't have more encounters in this timestep
//                         break 
//     else:
//         # if False, engage the double loop, as this N is too small to make the up-front sort of the sweep algorithm worthwhile
//
//         num_poss_ints = 0
//         num_encounters = 0
//         if len(circ_prograde_population_indices) > 0:
//             for i, circ_idx in enumerate(circ_prograde_population_indices):
//                 for j, ecc_idx in enumerate(ecc_prograde_population_indices):
//                     if (disk_bh_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_bh_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
//                         # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
//                         # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
//                         temp_bin_mass = disk_bh_pro_masses[circ_idx] + disk_bh_pro_masses[ecc_idx]
//                         bh_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
//                         mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
//                         prob_orbit_overlap = (1./np.pi)*mass_ratio_factor
//                         prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
//                         if prob_enc_per_timestep > 1:
//                             prob_enc_per_timestep = 1
//                         if chance_of_enc[i][j] < prob_enc_per_timestep:
//                             num_encounters = num_encounters + 1
//                             # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
//                             # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
//                             # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
//                             if disk_bh_pro_orbs_ecc[circ_idx] <= disk_bh_pro_orb_ecc_crit:
//                                 disk_bh_pro_orbs_ecc[circ_idx] = delta_energy_strong
//                                 disk_bh_pro_orbs_a[circ_idx] = disk_bh_pro_orbs_a[circ_idx]*(1.0 + delta_energy_strong)
//                                 # Catch for if orb_a > disk_radius_outer
//                                 if (disk_bh_pro_orbs_a[circ_idx] > disk_radius_outer):
//                                     disk_bh_pro_orbs_a[circ_idx] = disk_radius_outer - epsilon[i][j]
//                                 disk_bh_pro_orbs_ecc[ecc_idx] = disk_bh_pro_orbs_ecc[ecc_idx]*(1 - delta_energy_strong)
//                                 disk_bh_pro_orbs_a[ecc_idx] = disk_bh_pro_orbs_a[ecc_idx]*(1 - delta_energy_strong)
//                         num_poss_ints = num_poss_ints + 1
//                 num_poss_ints = 0
//                 num_encounters = 0
//
//     # Check finite
//     assert np.isfinite(disk_bh_pro_orbs_a).all(), \
//         "Finite check failed for disk_bh_pro_orbs_a"
//     assert np.isfinite(disk_bh_pro_orbs_ecc).all(), \
//         "Finite check failed for disk_bh_pro_orbs_ecc"
//     assert np.all(disk_bh_pro_orbs_a < disk_radius_outer), \
//         "disk_bh_pro_orbs_a contains values greater than disk_radius_outer"
//     assert np.all(disk_bh_pro_orbs_a > 0), \
//         "disk_bh_pro_orbs_a contains values <= 0"
//
//     return (disk_bh_pro_orbs_a, disk_bh_pro_orbs_ecc)





#[allow(clippy::collapsible_if)]
pub fn _circ_prograde_stars<'py>(
    py: Python<'py>,
    smbh_mass: f64,
    solar_mass: f64,
    bin_mass_1: PyReadonlyArray1<'py, f64>,
    bin_id_nums: PyReadonlyArray1<'py, f64>,
    bin_masses: PyReadonlyArray1<'py, f64>,
    bin_sep: PyReadonlyArray1<f64>,
    bin_ecc: PyReadonlyArray1<f64>,
    bin_orb_ecc: PyReadonlyArray1<f64>,
    bin_orb_a: PyReadonlyArray1<'py, f64>,
    bin_contact_sep: PyReadonlyArray1<'py, f64>,
    bin_hill_sep: PyReadonlyArray1<'py, f64>,
    epsilon_orb_a: PyReadonlyArray1<'py, f64>,
    bin_orbits_per_timestep: PyReadonlyArray1<'py, f64>,
    ecc_orb_max: PyReadonlyArray1<'py, f64>,
    ecc_orb_min: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_locations: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_eccentricities: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_masses: PyReadonlyArray1<'py, f64>,
    circ_prograde_population_id_nums: PyReadonlyArray1<'py, f64>,
    chances: PyReadonlyArray2<'py, f64>,
    bin_velocities: PyReadonlyArray1<'py, f64>,
    circ_velocities: PyReadonlyArray1<'py, f64>,
    bin_binding_energy: PyReadonlyArray1<'py, f64>,
    de_strong: f64,
    delta_energy_strong: f64,
    disk_radius_outer: f64,
    epsilon: f64,
) -> (Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>) {
// ) -> Bound<'py, PyList> {

    let bin_sep_slice = unsafe {bin_sep.as_slice_mut().unwrap() };
    let bin_ecc_slice = unsafe {bin_ecc.as_slice_mut().unwrap() };
    let bin_orb_ecc_slice = unsafe {bin_orb_ecc.as_slice_mut().unwrap() };

    let circ_prograde_population_locations_slice = unsafe { circ_prograde_population_locations.as_slice_mut().unwrap() };
    let circ_prograde_population_eccentricities_slice = unsafe { circ_prograde_population_eccentricities.as_slice_mut().unwrap() };

    // todo: double check these don't need to be mutable
    // buckets for appending, lists, inefficient
    let id_nums_poss_touch = PyList::empty(py);
    let frac_rhill_sep = PyList::empty(py);
    let id_nums_ionized_bin = PyList::empty(py);
    let id_nums_merged_bin = PyList::empty(py);

    // double loop, potential for a sort+sweep here?
    for i in 0..bin_mass_1.len().unwrap() {
        for j in 0..circ_prograde_population_locations.len().unwrap() {
            if !id_nums_ionized_bin.contains(bin_id_nums.get(i).unwrap()).unwrap() && !id_nums_merged_bin.contains(bin_id_nums.get(i).unwrap()).unwrap() {
                if (1.0 - *bin_orb_ecc.get(i).unwrap()) * *bin_orb_a.get(i).unwrap() < *ecc_orb_max.get(j).unwrap() && (1.0 + bin_orb_ecc.get(i).unwrap() * bin_orb_a.get(i).unwrap() > *ecc_orb_min.get(j).unwrap()) {
                    // temp_bin_mass / (3.0 * smbh_mass)
                    let bh_smbh_mass_ratio = (bin_masses.get(i).unwrap() + circ_prograde_population_masses.get(j).unwrap())/(3.0 * smbh_mass);
                    let prob_enc_per_timestep = ((1.0/PI) * (bh_smbh_mass_ratio.powf(1.0/3.0)) * bin_orbits_per_timestep.get(i).unwrap()).clamp(-100.0, 1.0);

                    // double check this syntax is right
                    let chance_of_encounter = chances.get([i, j]).unwrap();
                    if *chance_of_encounter < prob_enc_per_timestep {
                        let rel_vel_ms = (bin_velocities.get(i).unwrap() - circ_velocities.get(j).unwrap()).abs();
                        let ke_interloper = 0.5 * circ_prograde_population_masses.get(j).unwrap() * solar_mass * (rel_vel_ms.powi(2));
                        let hard = bin_binding_energy.get(i).unwrap() - ke_interloper;
                        if hard > 0.0 {
                            // these need to be mutable slices
                            bin_sep_slice[i] *= 1.0 - de_strong;
                            bin_ecc_slice[i] *= 1.0 + de_strong;
                            bin_orb_ecc_slice[i] *= 1.0 + delta_energy_strong;

                            // changing the interloper parameters
                            // this might be troublesome
                            circ_prograde_population_locations_slice[j] *= 1.0 + delta_energy_strong;
                            if circ_prograde_population_locations_slice[j] > disk_radius_outer {
                                circ_prograde_population_locations_slice[j] = disk_radius_outer - epsilon_orb_a.get(j).unwrap();
                            }
                            circ_prograde_population_eccentricities_slice[j] *= 1.0 + delta_energy_strong;
                            if bin_sep.get(i).unwrap() <= bin_contact_sep.get(i).unwrap() {
                                let _ = id_nums_merged_bin.append(*bin_id_nums.get(i).unwrap());
                            }
                        } else if hard < 0.0 {
                            bin_sep_slice[i] *= 1.0 - delta_energy_strong;
                            bin_ecc_slice[i] *= 1.0 + delta_energy_strong;
                            bin_orb_ecc_slice[i] *= 1.0 + delta_energy_strong;

                            circ_prograde_population_locations_slice[j] *= 1.0 - delta_energy_strong;
                            if circ_prograde_population_locations_slice[j] > disk_radius_outer {
                                circ_prograde_population_locations_slice[j] = disk_radius_outer - epsilon_orb_a.get(j).unwrap();
                            }
                            circ_prograde_population_eccentricities_slice[j] *= 1.0 - delta_energy_strong;

                            if bin_sep_slice[i] > *bin_hill_sep.get(i).unwrap() {
                                let _ = id_nums_ionized_bin.append(*bin_id_nums.get(i).unwrap());
                            }
                        }
                        // todo... not quite identical behavior, but prob fine??
                        // nah, make it identical
                        if *bin_ecc.get(i).unwrap() > 1.0 {
                            bin_ecc_slice[i] = 1.0 - epsilon;
                        }
                        if *bin_orb_ecc.get(i).unwrap() > 1.0 {
                            bin_orb_ecc_slice[i] = 1.0 - epsilon;
                        }
                        if *circ_prograde_population_eccentricities.get(i).unwrap() > 1.0 {
                            circ_prograde_population_eccentricities_slice[i] = 1.0 - epsilon;
                        }

                        let separation = (circ_prograde_population_locations.get(j).unwrap() - bin_orb_a.get(i).unwrap()).abs();
                        // perform a weighted average
                        // double check that this is equivalent to 
                        // center_of_mass = np.average([circ_prograde_population_locations[j], bin_orb_a[i]],
                        //                             weights=[circ_prograde_population_masses[j], bin_masses[i]])
                        let center_of_mass = ((circ_prograde_population_locations.get(j).unwrap() * circ_prograde_population_masses.get(j).unwrap()) + (bin_orb_a.get(i).unwrap() * bin_masses.get(i).unwrap())) / (circ_prograde_population_masses.get(j).unwrap() + bin_masses.get(i).unwrap());
                        let rhill_poss_encounter = center_of_mass * ((circ_prograde_population_masses.get(j).unwrap() + bin_masses.get(i).unwrap()) / (3. * smbh_mass)).powf(1.0/3.0);

                        if separation - rhill_poss_encounter < 0.0 {
                            let _ = id_nums_poss_touch.append(vec![circ_prograde_population_id_nums.get(j).unwrap(), bin_id_nums.get(i).unwrap()]);
                            let _ = frac_rhill_sep.append(separation/rhill_poss_encounter);
                        }
                    }
                }
            }
        }
    }

    // we only need to return these here, the others we've mutated in place
    (id_nums_poss_touch, frac_rhill_sep, id_nums_ionized_bin, id_nums_merged_bin)
}






