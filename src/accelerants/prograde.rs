use std::{collections::BTreeSet, f64::consts::PI};

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
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
    let time_factor = PI * (2.0e30 * smbh_mass * G_SI) / (C_SI.powi(3) * 3.15e7);
    let n_orbs_per_timestep: Vec<f64> = circ.iter().map(|p| {
        let t_orb = p.orb_a.powf(1.5) * time_factor;
        timestep_duration_yr / t_orb
    }).collect();

    // ── 4. Epsilon: Hill-radius × random — shape (N_circ, N_ecc) ─────────────
    // hill_radii[i] = disk_radius_outer * (m_circ / (3*(m_circ + M_smbh)))^(1/3)
    let hill_radii: Array1<f64> = Array1::from_iter(
        circ.iter().map(|p| {
            disk_radius_outer * (p.mass / (3.0 * (p.mass + smbh_mass))).cbrt()
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
                    let mass_ratio_factor  = bh_smbh_mass_ratio.cbrt();
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





