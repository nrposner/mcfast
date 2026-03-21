use pyo3::{exceptions::PyTypeError, prelude::*};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use phf::{Map, phf_map};

use crate::accelerants::{C_SI, EARTH_TO_SOL, FloatArray1, G_SI, JUPITER_TO_SOL, M_SUN_G, M_SUN_KG};

/// Calculate the gravitational radius r_g in SI units (meters)
/// This matches the Python si_from_r_g function behavior
pub fn si_from_r_g(smbh_mass: f64, distance_r_g: f64) -> f64 {
    // smbh_mass is in solar masses, convert to kg
    let smbh_mass_kg = smbh_mass * M_SUN_KG;
    
    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);
    
    // Calculate distance in meters
    distance_r_g * r_g
}


#[pyfunction]
/// Calculates the SI distance in meters from the mass and distance_r_g of an SMBH using
///
/// distance_r_g * (G_SI * mass) / (C_SI * C_SI)
///
/// `mass` must be a single value (scalar or 1-array) with AstroPy units or unitless and
/// denominated in solar masses
/// `distance_r_g` may be a Quantity array of AstroPy units, a unitless NumPy array, or a unitless scalar. In the absece of units, distance_r_g is assumed to be denominated in gravitational radii
pub fn si_from_r_g_helper<'py>(
    py: Python<'py>, 
    smbh_mass: &Bound<'_, PyAny>, 
    distance_r_g: &Bound<'_, PyAny>, // either unitless scalar or unitless array
) -> PyResult<FloatArray1<'py>> {
    // take the mass, assumed to be a scalar denominated in solar masses, and convert it to
    // kilograms
    let smbh_mass_kg = if let Ok(solarmass) = smbh_mass.extract::<f64>() {
        // if there is no associated unit, assume mass is denominated in solar masses, transform
        // into kilograms for calculations
        solarmass * M_SUN_KG
    } else if let Ok(solarmasses) = smbh_mass.extract::<PyReadonlyArray1<f64>>() {
        // if there is no associated unit, assume mass is denominated in solar masses, transform
        // into kilograms for calculations
        // and we just need the first value, as this is meant to be a scalar anyway
        let solarmass = solarmasses.as_slice().unwrap().first().unwrap();
        solarmass * M_SUN_KG
    } else if let Ok(quantity) = extract_scalar_unit(smbh_mass) {
        // there is some unit, transform to solar masses...
        let coeff = match quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for si_from_r_g: {:?}", quantity.unit)
                )
            )
        }?;

        // ...then to kilograms
        let solarmass = quantity.value / coeff;
        solarmass * M_SUN_KG
        // is this a bit silly? Yes, but that's also what the original function did, and
        // it makes things a bit easier to understand
    } else if let Ok(array_quantity) = extract_array_unit(smbh_mass) {
        // there is some unit, transform to solar masses...
        let coeff = match array_quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for si_from_r_g: {:?}", array_quantity.unit)
                )
            )
        }?;

        // ...then to kilograms
        // since this is meant to be a scalar, we only want to take the first element, 
        let solarmass = array_quantity.values.as_slice().unwrap().first().unwrap() / coeff;
        solarmass * M_SUN_KG
        // is this a bit silly? Yes, but that's also what the original function did, and
        // it makes things a bit easier to understand
    } else {
        return Err(
            PyTypeError::new_err(
                "Invalid type for smbh_mass: must be either a unitless scalar or AstroPy Quantity scalar".to_string()
            )
        )
    };

    // Calculate r_g = G * M / c^2
    let r_g = (G_SI * smbh_mass_kg) / (C_SI * C_SI);

    // extract the type and unit of distance_r_g, which may be an AstroPy Quantity array, a
    // unitless NumPy numerical array, or a unitless scalar
    // in the first case, attempt to extract it into a value and 
    if let Ok(array_quantity) = extract_array_unit(distance_r_g) {
        // allocate the output array up front
        let out = unsafe {PyArray1::new(py, array_quantity.values.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // acquire the appropriate conversion coefficient to turn r_g into an SI 
        let coeff = match array_quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for si_from_r_g: {:?}", array_quantity.unit)
                )
            )
        }?;

        // calculate the value in SI meters
        array_quantity.values.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            let solmass = val / coeff;
            out_slice[i] = solmass * r_g;
        });

        Ok(out)
    // in the second case, it's a unitless numpy array
    } else if let Ok(masses) = distance_r_g.extract::<PyReadonlyArray1<f64>>() { 
        // allocate the output array up front
        let out = unsafe {PyArray1::new(py, distance_r_g.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // no unit, so just assume it's in solar masses already
        masses.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            out_slice[i] = val * r_g;
        });

        Ok(out)

    // scalar case, because Python hates us 
    } else if let Ok(mass) = distance_r_g.extract::<f64>() {
        // allocate the output array up front
        let out = mass * r_g;
        Ok(PyArray1::from_slice(py, &[out]))
    } else if let Ok(quantity) = extract_scalar_unit(distance_r_g) {
        let coeff = match quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for si_from_r_g: {:?}", quantity.unit)
                )
            )
        }?;

        let solmass = quantity.value / coeff;
        let out = solmass * r_g;
        Ok(PyArray1::from_slice(py, &[out]))
    } else {
        Err(
            PyTypeError::new_err(
                "Unsupported format for si_from_r_g_helper: distance_r_g must be an AstroPy Quantity, a numeric NumPy array, or a unitless scalar".to_string()
            )
        )
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

    if let Ok(array_quantity) = extract_array_unit(distance_r_g) {
        // allocate the output array up front
        let out = unsafe {PyArray1::new(py, array_quantity.values.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        // since at the moment we only accept meters
        match array_quantity.unit {
            Unit::Meter => { }
            _ => return Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for r_g_from_units: {:?}", array_quantity.unit)
                )
            )
        };

        array_quantity.values.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            out_slice[i] = val / r_g;
        });

        Ok(out)
    } else if let Ok(masses) = distance_r_g.extract::<PyReadonlyArray1<f64>>() { 
        // allocate the output array up front
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
    if let Ok(array_quantity) = extract_array_unit(mass) {
        // allocate the output array up front
        let out = unsafe {PyArray1::new(py, array_quantity.values.len().unwrap(), false)};
        let out_slice = unsafe {out.as_slice_mut().unwrap()};

        let coeff = match array_quantity.unit {
            Unit::Gram => { Ok(M_SUN_G) }
            Unit::Kilogram => { Ok(M_SUN_KG) },
            Unit::EarthMass => { Ok(EARTH_TO_SOL) },
            Unit::JupiterMass => { Ok(JUPITER_TO_SOL) },
            Unit::SolarMass => { Ok(1.0) },
            _ => Err(
                PyTypeError::new_err(
                    format!("Unsupported unit for r_schwarzschild_of_m: {:?}", array_quantity.unit)
                )
            )
        }?;

        array_quantity.values.as_slice().unwrap().iter().enumerate().for_each(|(i, val)| {
            let solmass = val / coeff;
            let r_sch = (2.0 * G_SI * solmass / (C_SI.powi(2))) * M_SUN_KG;
            out_slice[i] = r_sch;
        });

        Ok(out)
    // otherwise, assume it's a numpy float array
    } else if let Ok(masses) = mass.extract::<PyReadonlyArray1<f64>>() { 
        // allocate the output array up front
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

/// Assume that the value starts in solar masses
pub fn r_schwarzschild_of_m_local(mass: f64) -> f64 {
    // using kg_to_sol here is equivalent to using .to(u.m) 
    (2.0 * G_SI * mass / (C_SI.powi(2))) * M_SUN_KG
}

pub struct Quantity {
    pub value: f64,
    pub unit: Unit
}
pub struct ArrayQuantity<'py> {
    pub values: PyReadonlyArray1<'py, f64>,
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
        Err(
            PyTypeError::new_err("Could not parse unit {unit_str}")
        )
    }

    // else, we'll have to decompose and take it apart

    // todo: make more robust, decomposing composite 
    // dimensions into their proper representations
    
    // Ok(Quantity {value, unit})
}



pub fn extract_array_unit<'py>(ob: &Bound<'py, PyAny>) -> PyResult<ArrayQuantity<'py>> {
    // extract numerical value
    let values = ob.getattr("value")?.extract::<PyReadonlyArray1<f64>>()?;
    // extract unit value
    let unit_obj = ob.getattr("unit")?;

    // use python method to extract string representation
    let binding = unit_obj.call_method0("to_string")?;
    let unit_str: &str = binding.extract()?;

    // if we get a match on a simple type, return directly
    if let Ok(unit) = parse_unit(unit_str) {
        Ok(ArrayQuantity { values, unit })
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
            PyTypeError::new_err(
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
