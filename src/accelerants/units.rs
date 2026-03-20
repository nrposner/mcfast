use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use phf::{Map, phf_map};

use crate::accelerants::{C_SI, EARTH_TO_SOL, FloatArray1, G_SI, JUPITER_TO_SOL, M_SUN_G, M_SUN_KG};

/// Calculate the gravitational radius r_g in SI units (meters)
/// This matches the Python si_from_r_g function behavior
pub fn si_from_r_g(smbh_mass: f64, distance_rg: f64) -> f64 {
    // smbh_mass is in solar masses, convert to kg
    let smbh_mass_kg = smbh_mass * M_SUN_KG;
    
    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);
    
    // Calculate distance in meters
    distance_rg * r_g
}



#[pyfunction]
pub fn si_from_r_g_helper<'py>(
    py: Python<'py>, 
    smbh_mass: f64, 
    distance_r_g: &Bound<'_, PyAny>,
) -> PyResult<FloatArray1<'py>> {

    // need to extract the unit attribute, if it exists
    // and make sure it's in solar masses
    
    let smbh_mass_kg = smbh_mass * M_SUN_KG;

    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);

    if let Ok(quantity) = extract_array_unit(distance_r_g) {
        let out = unsafe {PyArray1::new(py, quantity.value.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        let temp = distance_r_g.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
        let distance_r_g = temp.as_slice().unwrap();

        let coeff = match quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported unit for si_from_r_g: {:?}", quantity.unit)
                )
            )
        }?;

        distance_r_g.iter().enumerate().for_each(|(i, val)| {
            let solmass = val / coeff;
            out_slice[i] = solmass * r_g;
        });

        Ok(out)
    } else if let Ok(masses) = distance_r_g.extract::<PyReadonlyArray1<f64>>() { 

        let out = unsafe {PyArray1::new(py, distance_r_g.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // no unit, so just assume it's in solar masses already
        masses.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            out_slice[i] = val * r_g;
        });

        Ok(out)

    // scalar case, because Python hates us 
    } else if let Ok(mass) = distance_r_g.extract::<f64>() {
        let out = mass * r_g;
        Ok(PyArray1::from_slice(py, &[out]))
    } else {
        panic!("What the hell?")
    }
}

// si_from_r_g and r_g_from_units are the same operation in reverse, we just need to accept
// different unit types and deal with them.
#[pyfunction]
pub fn r_g_from_units_helper<'py>(
    py: Python<'py>, 
    smbh_mass: f64, 
    distance_r_g: &Bound<'_, PyAny>,
) -> PyResult<FloatArray1<'py>> {

    let smbh_mass_kg = smbh_mass * M_SUN_KG;

    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);

    if let Ok(quantity) = extract_array_unit(distance_r_g) {
        let out = unsafe {PyArray1::new(py, quantity.value.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        let temp = distance_r_g.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
        let distance_r_g = temp.as_slice().unwrap();

        // since at the moment we only accept meters
        match quantity.unit {
            Unit::Meter => { }
            _ => return Err(
                pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported unit for r_g_from_units: {:?}", quantity.unit)
                )
            )
        };

        distance_r_g.iter().enumerate().for_each(|(i, val)| {
            out_slice[i] = val / r_g;
        });

        Ok(out)
    } else if let Ok(masses) = distance_r_g.extract::<PyReadonlyArray1<f64>>() { 

        let out = unsafe {PyArray1::new(py, distance_r_g.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // no unit, so just assume it's in solar masses already
        masses.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            out_slice[i] = val / r_g;
        });

        Ok(out)

    // scalar case, because Python hates us 
    } else if let Ok(quantity) = extract_scalar_unit(distance_r_g) {
        let solmass = match quantity.unit {
            Unit::Meter => {
                quantity.value
            },
            _ => panic!("Unsupported unit for r_g_from_units: {:?}", quantity.unit)
        };

        let out = solmass / r_g;

        Ok(PyArray1::from_slice(py, &[out]))

    } else if let Ok(mass) = distance_r_g.extract::<f64>() {
        let out = mass / r_g;
        Ok(PyArray1::from_slice(py, &[out]))
    } else {
        panic!("distance_r_g could not be extracted into a valid type")
    }
}


/// Scalar version of r_schwarzschild_of_m for python-facing... or should vector version be
/// default?? 
#[pyfunction]
pub fn r_schwarzschild_of_m_helper<'py>(py: Python<'py>, mass: &Bound<'_, PyAny>) -> PyResult<FloatArray1<'py>> {

    // extract the attribute if it exists, and if so, make sure it's denominated in solar masses
    if let Ok(quantity) = extract_array_unit(mass) {
        let out = unsafe {PyArray1::new(py, quantity.value.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        let temp = mass.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
        let mass = temp.as_slice().unwrap();

        let coeff = match quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                pyo3::exceptions::PyValueError::new_err(
                    format!("Unsupported unit for r_schwarzschild_of_m: {:?}", quantity.unit)
                )
            )
        }?;

        mass.iter().enumerate().for_each(|(i, val)| {
            let solmass = val / coeff;
            let r_sch = (2.0 * G_SI * solmass / (C_SI.powi(2))) * M_SUN_KG;
            out_slice[i] = r_sch;
        });

        Ok(out)
    // otherwise, assume it's a numpy float array
    } else if let Ok(masses) = mass.extract::<PyReadonlyArray1<f64>>() { 

        let out = unsafe {PyArray1::new(py, masses.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // no unit, so just assume it's in solar masses already
        masses.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            let r_sch = (2.0 * G_SI * val / (C_SI.powi(2))) * M_SUN_KG;
            out_slice[i] = r_sch;
        });

        Ok(out)
    // otherwise, we've received an invalid input type
    } else {
        panic!("Unsupported input type for r_schwarzschild_of_m: mass input must be an array")
    }
}

pub struct Quantity {
    pub value: f64,
    pub unit: Unit
}
pub struct ArrayQuantity<'py> {
    pub value: PyReadonlyArray1<'py, f64>,
    pub unit: Unit
}

pub fn extract_scalar_unit(ob: &Bound<'_, PyAny>) -> PyResult<Quantity> {
    // extract numerical value
    let value: f64 = ob.getattr("value")?.extract()?;
    // extract unit value
    let unit_obj = ob.getattr("unit")?;

    // use python method to extract string representation
    let binding = unit_obj.call_method0("to_string")?;
    let unit_str: &str = binding.extract()?;

    // if we get a match on a simple type, return directly
    if let Ok(unit) = parse_unit(unit_str) {
        Ok(Quantity { value, unit })
    } else {
        panic!("Could not parse unit {unit_str}")
    }

    // else, we'll have to decompose and take it apart

    // todo: make more robust, decomposing composite 
    // dimensions into their proper representations
    
    // Ok(Quantity {value, unit})
}



pub fn extract_array_unit<'py>(ob: &Bound<'py, PyAny>) -> PyResult<ArrayQuantity<'py>> {
    // extract numerical value
    let value = ob.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
    // extract unit value
    let unit_obj = ob.getattr("unit")?;

    // use python method to extract string representation
    let binding = unit_obj.call_method0("to_string")?;
    let unit_str: &str = binding.extract()?;

    // if we get a match on a simple type, return directly
    if let Ok(unit) = parse_unit(unit_str) {
        Ok(ArrayQuantity { value, unit })
    } else {
        panic!("Could not parse unit {unit_str}")
    }

    // else, we'll have to decompose and take it apart

    // todo: make more robust, decomposing composite 
    // dimensions into their proper representations

    // Ok(Quantity {value, unit})
}

pub static UNIT_MAP: Map<&'static str, Unit> = phf_map! {
    // length
    "m" | "meter" => Unit::Meter,
    "cm" | "centimeter" => Unit::Centimeter,
    "au" | "AU" | "astronomical_unit" => Unit::AU,
    "earthRad" | "R_earth" | "Rearth" => Unit::EarthRad,
    "jupiterRad" | "R_jup" | "Rjup" | "R_jupiter" | "Rjupiter" => Unit::JupiterRad,
    "solRad" | "R_sun" | "Rsun" => Unit::SolarRad,
    "a0" => Unit::BohrRad,

    // mass
    "kg" | "kilogram" => Unit::Kilogram,
    "g" | "gram" => Unit::Gram,
    "t" | "tonne" => Unit::MetricTon,
    "me" | "M_e" => Unit::ElectronMass,
    "mp" | "M_p" => Unit::ProtonMass,
    "u" | "da" | "Dalton" => Unit::Dalton,
    "earthMass" | "M_earth" | "Mearth" | "geoMass" | "Mgeo" => Unit::EarthMass,
    "jupiterMass" | "M_jup" | "Mjup" | "M_jupiter" | "Mjupiter" | "jovMass" => Unit::JupiterMass,
    "solMass" | "M_sun" | "Msun" => Unit::SolarMass,

    // imperial mass units
    "oz" | "ounce" => Unit::Oz,
    "lb" | "lbm" | "pound" => Unit::Lb,
    "slug" => Unit::Slug,
    "st" | "stone" => Unit::Stone,
    "ton" => Unit::ImperialTon,

};

pub fn parse_unit(unit_str: &str) -> PyResult<Unit> {
    // for really common units, go directly to byte comparison
    match unit_str.as_bytes() {
        b"kg" | b"kilogram" => return Ok(Unit::Kilogram),
        b"t" | b"tonne" => return Ok(Unit::MetricTon),
        b"earthMass" | b"M_earth" => return Ok(Unit::EarthMass),
        b"jupiterMass" | b"M_jup" => return Ok(Unit::JupiterMass),
        b"solMass" | b"Msun" => return Ok(Unit::SolarMass),
        b"m" | b"meter" => return Ok(Unit::Meter),

        _ => {}
    }

    // otherwise fall back to perfect hash
    // should we remove the common units from the hash?
    UNIT_MAP.get(unit_str)
        .copied() // or .cloned() if UnitEnum doesn't impl Copy
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported unit: {}", unit_str)
            )
        })
}

#[derive(Clone, Copy, Debug)]
pub enum Unit {
    // Length
    /// Fundamental SI unit of Length
    Meter,
    /// 0.01 Meters, Fundamental CGS unit of Length
    Centimeter,
    // /// 1000 Meters
    // Kilometer,
    /// 1.4959787e+11 Meters, approximately the mean Earth-Sun distance
    AU, 
    /// 6378100 Meters, Earth Radius
    EarthRad,
    ///71492000 Meters, Jupiter Radius
    JupiterRad,
    /// 6.957e+8 Meters, Solar Radius
    SolarRad,
    /// 5.2917721e-11 Meters, Bohr Radius
    BohrRad,

    // SI Mass
    /// Fundamental SI unit of Mass
    Kilogram,
    /// 0.001 Kg
    Gram,
    /// 1000 Kg
    MetricTon,
    /// 9.1093837e-31 Kilograms
    ElectronMass,
    /// 1.6726219e-27 Kilograms
    ProtonMass,
    /// 1.6605391e-27 Kilograms, aliased Da, U, the unified atomic mass unit
    Dalton,
    /// 5.9721679e+24 Kilograms
    EarthMass, 
    /// 1.8981246e+27 Kilograms
    JupiterMass,
    /// 1.9884099e+30 Kilograms
    SolarMass,

    // Imperial Mass
    /// 28.349523 Grams, the international avourdupois ounce
    Oz,
    /// 16 Ozs, the international avourdupois ounce
    Lb,
    /// 32.174049 Lbs
    Slug,
    /// 14 Lbs, the international avourdupois stone
    Stone,
    /// 2000 Lbs, the international avourdupois ton 
    ImperialTon
}
