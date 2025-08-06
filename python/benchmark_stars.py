import time
import sys
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy

import mcfacts_helper as helper

SINGLETON_PROP = 0.15
AXIS_TOLERANCE =  0.0000001
ECCENTRICITY_TOLERANCE = 0.0001 # the eccentricities are much less reliable than the axes
SPEED_TOLERANCE = 0.80

def components_from_EL(E, L, units='geometric', smbh_mass=1e8):
    """Calculates new orb_a and eccentricity from specific energy and specific angular momentum

    Parameters
    ----------
    E : float
        specific energy (per unit mass)
    L : float
        specific angular momentum (per unit mass)
    units : str, optional
        whether to use geometric units, by default 'geometric'
    smbh_mass : float, optional
        Mass [Msun] of the SMBH, by default 1e8

    Returns
    -------
    orb_a, ecc
        new orb_a [r_{g,SMBH}] and ecc values
    """
    # takes in SPECIFIC E, L (should be lower case variables)
    G_val = 1
    if units != 'geometric':
        G_val = const.G.si.value
    # compute a, e from E,L
    #
    orb_a = - (G_val * smbh_mass)/(2*E)
    one_minus_ecc2_sqrt = L/np.sqrt(G_val * smbh_mass * orb_a)
    # Hack, deal with roundoff error!
    if one_minus_ecc2_sqrt - 1 > 0 and one_minus_ecc2_sqrt - 1 < 1e-2:
        one_minus_ecc2_sqrt = 1-1e-2
    if one_minus_ecc2_sqrt > 1:
        raise Exception(" Impossible eccentricity value, based on  ", one_minus_ecc2_sqrt)
    with np.errstate(invalid="ignore"):
        ecc = np.sqrt(1-one_minus_ecc2_sqrt**2)
    return orb_a / (2 * smbh_mass), ecc


def cubic_y_root(x0, y0, sanity=False):
    """Calculate the root of cubic function f(y) = x0*y^3 + 1.5*y - y0

    Parameters
    ----------
    x0 : float
        dimensionless x0 variable
    y0 : float
        dimensionless y0 variable
    sanity : bool, optional
        Switch, turns on extra print statements, by default False

    Returns
    -------
    roots : 
        roots of the function
    """
    coefficients = np.array([x0, 0, +1.5, -y0])
    poly = np.polynomial.Polynomial(coefficients[::-1])
    roots = poly.roots()
    if sanity:
        print(" Root sanity check ", roots)
        yval = roots[0]
        val0 = x0 * yval ** 3 + 1.5 * yval - y0
        print(yval, val0)
    return roots

def cubic_y_root_cardano(x0, y0, sanity=False):
    """
    Optimized version of cubic_y_root using an analytic solver.
    Solves the equation: x0*y^3 + 1.5*y - y0 = 0
    """
    # handle the edge case where x0 is zero, becomes 1.5*y - y0 = 0
    if x0 == 0:
        return np.array([y0 / 1.5])

    # convert to the standard depressed cubic form y^3 + p*y + q = 0
    # by dividing the original equation by the leading coefficient x0
    p = 1.5 / x0
    q = -y0 / x0

    # calculate the discriminant term to see if there will be one or three real roots
    delta = (q/2)**2 + (p/3)**3

    if delta >= 0:
        # discriminant positive or 0, one real root, two complex roots
        sqrt_delta = np.sqrt(delta)
        u = np.cbrt(-q/2 + sqrt_delta)
        v = np.cbrt(-q/2 - sqrt_delta)
        roots = np.array([u + v])
    else:
        # discriminant negative, three real roots
        term1 = 2 * np.sqrt(-p / 3)
        phi = np.arccos((3 * q) / (p * term1))
        
        y1 = term1 * np.cos(phi / 3)
        y2 = term1 * np.cos((phi + 2 * np.pi) / 3)
        y3 = term1 * np.cos((phi + 4 * np.pi) / 3)
        roots = np.array([y1, y2, y3])
    
    if sanity:
        print(" Root sanity check ", roots)
        yval = roots[0]
        val0 = x0 * yval**3 + 1.5 * yval - y0
        print(yval, val0)
        
    return roots

def cubic_finite_step_root(x0, y0, OmegaS, sanity=False):
    """Determine allowed finite step size

    Parameters
    ----------
    x0 : float
        dimensionless x0 value
    y0 : float
        dimensionless y0 value
    OmegaS : float
        Orbital frequency [???]
    sanity : bool, optional
        Switch, turns on extra print statements, by default False

    Returns
    -------
    roots_x,roots_y : np.array
        Found roots
    """

    # y**3 + (2(x_0 - OmegaS * y_0)) * y + 2 * OmegaS
    coefficients = np.array([1, 0, 2 * (x0 - OmegaS * y0), 2 * OmegaS])
    poly = np.polynomial.Polynomial(coefficients)
    roots_y = poly.roots()

    # after finding all the real roots of y, then find the corresponding value for x for each one

    # now map to solutions for x!
    roots_x = -1 / (2 * roots_y ** 2)
    indx_ok = np.logical_and(roots_y > 0, np.abs(roots_x) < 1e10)
    indx_ok = np.logical_and(indx_ok, roots_x < 0)  # only pick roots that are bound
    roots_x = roots_x[indx_ok]
    roots_y = roots_y[indx_ok]
    if sanity:
        print(" Finite stepsize sanity check, both should be zero; second is trivial ")
        print(roots_y - y0 - (roots_x - x0) / OmegaS, roots_x + 1. / (2 * roots_y ** 2))
    return np.c_[roots_x, roots_y]


def cubic_finite_step_root_cardano(x0, y0, OmegaS, sanity = False):
    """
    Optimized version to determine allowed finite step size using an
    analytic solution for the depressed cubic equation.
    """

    # we have a polynomial y**3 + (2(x_0 - OmegaS * y_0)) * y + 2 * OmegaS 
    # which we will summarize as y**3 + p*y + q = 0
    # it's a depressed cubic root, with no square term

    p = 2 * (x0 - OmegaS * y0)
    q = 2 * OmegaS

    if p == 0: 
        roots_y = np.array([np.cbrt(-q)]) # just return the cube root of -2*OmegaS
    else:
        # calculate the discriminant term to see if there will be one or three real roots
        delta = (q/2)**2 + (p/3)**3

        if delta >= 0:
            # discriminant positive or 0, one real root, two complex roots
            sqrt_delta = np.sqrt(delta)
            u = np.cbrt(-q/2 + sqrt_delta)
            v = np.cbrt(-q/2 - sqrt_delta)
            roots_y = np.array([u + v]) # The only real root
        else:
            # discriminant negative, three real roots
            # this is more numerically stable than the standard Cardano formula for this case
            term1 = 2 * np.sqrt(-p / 3)
            phi = np.arccos( (3 * q) / (p * term1) ) # simplified from (3q)/(2p*sqrt(-p/3))

            y1 = term1 * np.cos(phi / 3)
            y2 = term1 * np.cos((phi + 2 * np.pi) / 3)
            y3 = term1 * np.cos((phi + 4 * np.pi) / 3)
            roots_y = np.array([y1, y2, y3])

    with np.errstate(divide='ignore'): # ignore division by zero warnings for invalid roots
        roots_x = -1 / (2 * roots_y ** 2)

    # filter for valid, physical roots
    indx_ok = np.logical_and(roots_y > 0, np.isfinite(roots_x))
    indx_ok = np.logical_and(indx_ok, roots_x < 0)  # only pick roots that are bound

    roots_x = roots_x[indx_ok]
    roots_y = roots_y[indx_ok]

    if sanity:
        print(" Finite stepsize sanity check, both should be zero; second is trivial ")
        print(roots_y - y0 - (roots_x - x0) / OmegaS, roots_x + 1. / (2 * roots_y ** 2))

    return np.c_[roots_x, roots_y] 



def transition_physical_as_EL(E1, L1, E2, L2, DeltaE, m1, m2, units='geometric', smbh_mass=1e8, sanity=False, cardano = False):
    """Calculates final energy and angular momentum states

    Parameters
    ----------
    E1 : float
        energy of object 1
    L1 : float
        angular momentum of object 1
    E2 : float
        energy of object 2
    L2 : float
        angular momentum of object 2
    DeltaE : float
        change in energy
    m1 : float
        mass of object 1
    m2 : float
        mass of object 2
    units : str, optional
        Switch to control type of units, by default 'geometric'
    smbh_mass : float, optional
        SMBH mass, by default 1e8
    sanity : bool, optional
        Switch to turn on extra print statements, by default True
    """
    G_val = 1
    if units != 'geometric':
        G_val = const.G.si.value

    # Assume consistent units SI only
    eps1 = E1 / m1
    eps2 = E2 / m2
    ell1 = L1/m1
    ell2 = L2/m2

    # Find Omega0 scale, which is based on the 'acceptor' (2) non-eccentric object. This means ell0 = ell2
    ell0 = ell2
    Omega0 = (G_val * smbh_mass) ** 2 / ell0 ** 3
    eps0 = ell0 * Omega0

    if sanity:
        # In case we need them, compute the frequencies of the other two objects
        eps1_f = (E1 + DeltaE) / m1
        eps2_f = (E2 - DeltaE) / m2
        Omega1 = np.sqrt(-2 * eps1) ** 3 / (G_val * smbh_mass)
        # final frequencies of the objects
        Omega1_f = np.sqrt(-2 * eps1_f) ** 3 / (G_val * smbh_mass)
        Omega2_f = np.sqrt(-2 * eps2_f) ** 3 / (G_val * smbh_mass)

        print(" Dimensionless frequencies; second should be nearly unity if circular", Omega1 / Omega0, Omega2 / Omega0)
        print(" Dimensionless final frequencies;", Omega1_f / Omega0, Omega2_f / Omega0)

    Omega2 = np.sqrt(-2 * eps2) ** 3 / (G_val * smbh_mass)

    # Dimensionless variables
    x0 = eps2 / eps0  # close to -1/2
    y0 = ell2 / ell0   # 1, by construction
    x0_alt = eps1 / eps0
    y0_alt = ell1 / ell0

    # depending on sign of Delta E, pick which choice of Omega0 we use.
    Omega_star = np.inf
    if DeltaE * (E2 - E1) < 0:
        if sanity:
            print(" Contraction ")
        # Contraction scenario: the two objects move closer together in energy
        #  - case 1: object 2 is in a more bound orbit (E2-E1 <0) and Delta E>0
        #  - case 2: object 2 is in a less bound orbit (E2-E1>0) but DeltaE <0
        # In this case, we can have one or both objects intersect the forbidden region

        # Determine the finite step size allowed.
        # NON-GENERAL ASSUMPTION FOR SIMPLICITY: Assume we are contracting on object 2! So not quite generic. Pick the other if it is object 1
        # But the stepsize constraint is set by object 1 (x0_alt), intersecting the boundary
        #   -  Object 1 is moving to tighter orbits (lower energy magnitude), so root of x is increasing in magnitude!
        Omega_trial = Omega2
        if cardano:
            # my_stepsize_roots = cubic_finite_step_root_cardano(x0_alt, y0_alt, Omega_trial / Omega0)
            roots_buffer, valid_count = helper.cubic_finite_step_root_cardano(x0_alt, y0_alt, Omega_trial / Omega0)
            all_roots = np.array(roots_buffer)
            my_stepsize_roots = all_roots[:valid_count]

        else: 
            my_stepsize_roots = cubic_finite_step_root(x0_alt, y0_alt, Omega_trial / Omega0)

        if sanity:
            print(" Pick root n between : x", my_stepsize_roots[:, 0], "between ", (x0, x0_alt), ", y ", my_stepsize_roots[:, 1],  " between ", (y0, y0_alt))
        my_stepsize_roots = my_stepsize_roots[my_stepsize_roots[:, 0] < x0]  # pick in between the two initial points
        if sanity:
            print("Stepsizing root confirmation: x", my_stepsize_roots[:, 0], (x0_alt, x0), " y :",  my_stepsize_roots[:, 1], (y0_alt, y0))
        # pick the root in between the two
        Omega_star = Omega_trial
        if len(my_stepsize_roots) != 0:
            DeltaE_max = m1 * (my_stepsize_roots[0, 0] - x0_alt) * eps0  # largest possible stepsize, note this is a *specific* energy, and applied to object 1
            if sanity:
                print(" Energy step compared to stepsize limit ", DeltaE, DeltaE_max)
            if np.abs(DeltaE) > np.abs(DeltaE_max):
                if sanity:
                    print(" Stepsize limit applied !")
                DeltaE = DeltaE_max
            else:
                if sanity:
                    print(" Don't reach the boundary - fine ! ")
    else:
        if sanity:
            print(" Expansion ")
            print("Dimensionless root finder: coordinates (should be close to -1/2, 1)", x0, y0)

        # Slope calculation, based on object 2 ('accepting' object/circular case)
        if cardano:
            ret = helper.cubic_y_root_cardano(x0, y0)
            my_roots = np.array(ret[0][0:ret[1]])
        else:
            my_roots = cubic_y_root(x0, y0)
        # restore physical units, these are y values; ell = y*ell0; and \Omega = (GM)^2/ell^3
        my_roots_ell = ell0 * my_roots
        my_roots_omega = (G_val * smbh_mass) ** 2 / my_roots_ell ** 3  # note order reversal!
        my_roots_omega.sort()

        if sanity:
            print(" Raw dimensionless roots",  my_roots_omega/Omega0)
            print(" Eliminate roots with large imaginary part")
        indx_ok = np.abs(np.imag(my_roots_omega/Omega0)) < 1e-3
        my_roots_omega = my_roots_omega[indx_ok]
        indx_ok = np.real(my_roots_omega) > 0  # don't change direction - one has another sign
        my_roots_omega = my_roots_omega[indx_ok]
        if sanity:
            print("Remaining dimensionless roots", my_roots_omega/Omega0)

        if sanity:
            print(" Dimensionless root finder part 2: coordinates for eccentric system ", x0_alt, y0_alt)

        if cardano:
            ret = helper.cubic_y_root_cardano(x0_alt, y0_alt)
            my_roots_alt = np.array(ret[0][0:ret[1]])
        else:
            my_roots_alt = cubic_y_root(x0_alt, y0_alt)

        my_roots_omega_alt = (G_val * smbh_mass) ** 2/(ell0 * my_roots_alt) ** 3
        my_roots_omega_alt = np.real(my_roots_omega_alt[np.real(my_roots_omega_alt) > 0])
        my_roots_omega_alt.sort()

        if sanity:
            print(" Omega values for both tangents ", my_roots_omega/Omega0, my_roots_omega_alt/Omega0)

        # Non-contracting scenario, the two objects move away. We can use any omega smaller than the largest root above
        Omega_star = np.min(np.real(np.concatenate((my_roots_omega, my_roots_omega_alt))))

    #DeltaE = np.sqrt(m1 * m2 / (m1 + m2)**2) * DeltaE
    # Transition
    DeltaL = DeltaE / Omega_star
#    DeltaL = DeltaE/Omega0  # use circular case
    return E1+DeltaE, E2-DeltaE, L1+DeltaL, L2-DeltaL


def encounters_new_orba_ecc(smbh_mass,
                            orb_a_give,
                            orb_a_take,
                            mass_give,
                            mass_take,
                            ecc_give,
                            ecc_take,
                            radius_give,
                            radius_take,
                            id_num_give,
                            id_num_take,
                            delta_energy_strong,
                            flag_obj_types,
                            cardano = False):
    """Calculate new orb_a and ecc values for two objects that dynamically interact

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    orb_a_give : float
        Semi-major axis [r_{g,SMBH}] of the object donating energy (typically the eccentric)
    orb_a_take : float
        Semi-major axis [r_{g,SMBH}] of the object accreting energy (typically the circular)
    mass_give : float
        Mass [M_sun] of the object donating energy
    mass_take : float
        Mass [M_sun] of the object accreting energy
    ecc_give : float
        Eccentricity of the object donating energy
    ecc_take : float
        Eccentricity of the object accreting energy
    radius_give : float
        Radius [r_{g,SMBH}] of the object donating energy
    radius_take : float
        Radius [r_{g,SMBH}] of the object accreting energy
    id_num_give : int
        ID number of the object donating energy
    id_num_take : int
        ID number of the object accreting energy
    delta_energy_strong : float
        Average energy change per strong encounter
    flag_obj_types : int
        Switch determining the type of interaction
        0 : eccentric star - circular star
        1 : eccentric black hole - circular star
        2 : eccentric black hole - circular black hole
        3 : eccentric star - eccentric star

    Returns
    -------
    orb_a_give_final : float
        New semi-major axis [r_{g,SMBH}] of the object donating energy
    orb_a_take_final : float
        New semi-major axis [r_{g,SMBH}] of the object accreting energy
    ecc_give_final : float
        New eccentricity of the object donating energy
    ecc_take_final : float
        New eccentricity of the object accreting energy
    id_num_unbound : int
        ID number of object unbound from the disk (if any, otherwise None)
    id_num_flipped_rotation : int
        ID number of object flipped from prograde to retrograde (if any, otherwise None)
    """
    # using units of 2M, concert to geometric units (G=1) using *solar mass units* for distance
    smbh_mass_geometric = 1
    mass_scale = smbh_mass / 1
    orb_a_give_geometric = orb_a_give * 2 * smbh_mass_geometric
    orb_a_take_geometric = orb_a_take * 2 * smbh_mass_geometric
    mass_give_geometric = mass_give / mass_scale
    mass_take_geometric = mass_take / mass_scale

    if flag_obj_types == 0:  # ecc star - circ star
        radius_give_geometric = radius_give * 2 * smbh_mass_geometric
        radius_take_geometric = radius_take * 2 * smbh_mass_geometric
        v_relative = np.sqrt(smbh_mass_geometric / orb_a_give_geometric) - np.sqrt(smbh_mass_geometric / orb_a_take_geometric)
        v_esc_sq = (smbh_mass_geometric / max(radius_give_geometric, radius_take_geometric))

    elif flag_obj_types == 1:  # ecc BH - circ star
        radius_give_geometric = 2 * mass_give_geometric
        radius_take_geometric = radius_take * 2 * smbh_mass_geometric
        # for BH the radius should be R = 2 M_BH = rg_unit * M_bh / M_smbh = 2 M_smbh * M_bh / M_smbh (in G = c = 1 units)
        v_relative = np.sqrt(smbh_mass_geometric / orb_a_give_geometric) - np.sqrt(smbh_mass_geometric / orb_a_take_geometric)
        v_esc_sq = 1  # for BH we want this to be 1

    E_give_initial = - mass_give_geometric * smbh_mass_geometric / (2 * orb_a_give_geometric)
    E_take_initial = - mass_take_geometric * smbh_mass_geometric / (2 * orb_a_take_geometric)
    J_give_initial = mass_give_geometric * np.sqrt(smbh_mass_geometric * orb_a_give_geometric * (1 - ecc_give**2))
    J_take_initial = mass_take_geometric * np.sqrt(smbh_mass_geometric * orb_a_take_geometric * (1 - ecc_take**2))

    mu_geometric = mass_give_geometric * mass_take_geometric / (mass_give_geometric + mass_take_geometric)
    Delta_E = delta_energy_strong * mu_geometric * (1 / ((1 / v_relative**2) + (1 / v_esc_sq)))

    id_num_unbound = None
    id_num_flipped_rotation = None

    if cardano:
        E_give_final, E_take_final, J_give_final, J_take_final = helper.transition_physical_as_el(E_give_initial, J_give_initial, E_take_initial, J_take_initial, Delta_E, mass_give_geometric, mass_take_geometric, None, smbh_mass=smbh_mass_geometric)

    else:
        E_give_final, E_take_final, J_give_final, J_take_final = transition_physical_as_EL(E_give_initial, J_give_initial, E_take_initial, J_take_initial, Delta_E, mass_give_geometric, mass_take_geometric, smbh_mass=smbh_mass_geometric, sanity=False, cardano = cardano)

    # if object is unbound, don't change parameters so they can be recorded
    # give object (typically eccentric) is unbound
    if E_give_initial + Delta_E > 0:
        orb_a_give_final = orb_a_give
        ecc_give_final = ecc_give
        id_num_unbound = id_num_give
        orb_a_take_final, ecc_take_final = components_from_EL(E_take_final / mass_take_geometric, J_take_final / mass_take_geometric, smbh_mass=smbh_mass_geometric)

    # take object (typically circular) is unbound
    elif E_take_initial - Delta_E > 0:
        orb_a_take_final = orb_a_take
        ecc_take_final = ecc_take
        id_num_unbound = id_num_take
        orb_a_give_final, ecc_give_final = components_from_EL(E_give_final / mass_give_geometric, J_give_final / mass_give_geometric, smbh_mass=smbh_mass_geometric)

    else:
        orb_a_give_final, ecc_give_final = components_from_EL(E_give_final / mass_give_geometric, J_give_final / mass_give_geometric, smbh_mass=smbh_mass_geometric)
        orb_a_take_final, ecc_take_final = components_from_EL(E_take_final / mass_take_geometric, J_take_final / mass_take_geometric, smbh_mass=smbh_mass_geometric)

    # give object is flipped from prograde to retrograde
    if J_give_final < 0:
        ecc_give_final = 0.0
        id_num_flipped_rotation = id_num_give
    # take object is flipped from prograde to retrograde
    elif J_take_final < 0:
        ecc_take_final = 0.0
        id_num_flipped_rotation = id_num_take

    return orb_a_give_final, orb_a_take_final, ecc_give_final, ecc_take_final, id_num_unbound, id_num_flipped_rotation

def r_g_from_units(smbh_mass, distance):
    """Calculate the SI distance from r_g

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of the SMBH
    distance_rg : astropy.units.quantity.Quantity
        Distances

    Returns
    -------
    distance_rg : numpy.ndarray
        Distances [r_g]
    """
    # Calculate c and G in SI
    c = const.c.to('m/s')
    G = const.G.to('m^3/(kg s^2)')
    # Assign units to smbh mass
    if hasattr(smbh_mass, 'unit'):
        smbh_mass = smbh_mass.to('solMass')
    else:
        smbh_mass = smbh_mass * u.solMass
    # convert smbh mass to kg
    smbh_mass = smbh_mass.to('kg')
    # Calculate r_g in SI
    r_g = G*smbh_mass/(c ** 2)
    # print(r_g)
    # Calculate distance
    distance_rg = distance.to("meter") / r_g

    # Check to make sure units are okay.
    assert u.dimensionless_unscaled == distance_rg.unit, \
        "distance_rg is not dimensionless. Check your input is a astropy Quantity, not an astropy Unit."
    assert np.isfinite(distance_rg).all(), \
        "Finite check failure: distance_rg"
    assert np.all(distance_rg > 0), \
        "Finite check failure: distance_rg"

    return distance_rg

def circular_singles_encounters_prograde_stars(
        smbh_mass,
        disk_star_pro_orbs_a,
        disk_star_pro_masses,
        disk_star_pro_radius,
        disk_star_pro_orbs_ecc,
        disk_star_pro_id_nums,
        rstar_rhill_exponent,
        timestep_duration_yr,
        disk_star_pro_orb_ecc_crit,
        delta_energy_strong_mu,
        delta_energy_strong_sigma,
        disk_radius_outer,
        rng_here,
        cardano = False
        ):
    """"Adjust orb ecc due to encounters between 2 single circ pro stars

    Parameters
    ----------
    smbh_mass : float
        Mass [M_sun] of supermassive black hole
    disk_bh_pro_orbs_a : numpy.ndarray
        Orbital semi-major axes [r_{g,SMBH}] of prograde singleton star at start of a timestep (math:`r_g=GM_{SMBH}/c^2`) with :obj:`float` type
    disk_bh_pro_masses : numpy.ndarray
        Masses [M_sun] of prograde singleton star at start of timestep with :obj:`float` type
    disk_star_pro_radius : numpy.ndarray
        Radii [Rsun] of prograde singleton star at start of timestep with :obj: `float` type
    disk_bh_pro_orbs_ecc : numpy.ndarray
        Orbital eccentricity [unitless] of singleton prograde star with :obj:`float` type
    disk_star_pro_id_nums : numpy.ndarray
        ID numbers of singleton prograde stars
    rstar_rhill_exponent : float
        Exponent for the ratio of R_star / R_Hill. Default is 2
    timestep_duration_yr : float
        Length of timestep [yr]
    disk_bh_pro_orb_ecc_crit : float
        Critical orbital eccentricity [unitless] below which orbit is close enough to circularize
    delta_energy_strong_mu : float
        Average energy change [units??] per strong encounter
    delta_energy_strong_sigma : float
        Standard deviation of average energy change per strong encounter

    Returns
    -------
    disk_star_pro_orbs_a : numpy.ndarray
        Updated BH semi-major axis [r_{g,SMBH}] perturbed by dynamics with :obj:`float` type
    disk_star_pro_orbs_ecc : numpy.ndarray
        Updated BH orbital eccentricities [unitless] perturbed by dynamics with :obj:`float` type
    disk_star_pro_id_nums_touch : numpy.ndarray
        ID numbers of stars that will touch each other

    Notes
    -----
    Return array of modified singleton star orbital eccentricities perturbed
    by encounters within :math:`f*R_{Hill}`, where f is some fraction/multiple of
    Hill sphere radius R_H

    Assume encounters between damped star (e<e_crit) and undamped star
    (e>e_crit) are the only important ones for now.
    Since the e<e_crit population is the most likely BBH merger source.

    1, find those orbiters with e<e_crit and their
        associated semi-major axes a_circ =[a_circ1, a_circ2, ..] and masses m_circ =[m_circ1,m_circ2, ..].

    2, calculate orbital timescales for a_circ1 and a_i and N_orbits/timestep. 
        For example, since
        :math:`T_orb =2\\pi \sqrt(a^3/GM_{smbh})`
        and
        .. math::
        a^3/GM_{smbh} = (10^3r_g)^3/GM_{smbh} = 10^9 (a/10^3r_g)^3 (GM_{smbh}/c^2)^3/GM_{smbh} \\
                    = 10^9 (a/10^3r_g)^3 (G M_{smbh}/c^3)^2 

        So
        .. math::
            T_orb   = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} GM_{smbh}/c^3 \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (6.7e-11*2e38/(3e8)^3) \\
                    = 2\\pi 10^{4.5} (a/10^3r_g)^{3/2} (13.6e27/27e24) \\
                    = \\pi 10^{7.5}  (a/10^3r_g)^{3/2} \\
                    ~ 3yr (a/10^3r_g)^3/2 (M_{smbh}/10^8M_{sun}) \\
        i.e. Orbit~3yr at 10^3r_g around a 10^8M_{sun} SMBH.
        Therefore in a timestep=1.e4yr, a BH at 10^3r_g orbits the SMBH N_orbit/timestep =3,000 times.

    3, among population of orbiters with e>e_crit,
        find those orbiters (a_i,e_i) where a_i*(1-e_i)< a_circ1,j <a_i*(1-e_i) for all members a_circ1,j of the circularized population 
        so we can test for possible interactions.

    4, calculate mutual Hill sphere R_H of candidate binary (a_circ1,j ,a_i).

    5, calculate ratio of 2R_H of binary to size of circular orbit, or (2R_H/2pi a_circ1,j)
        Hill sphere possible on both crossing inwards and outwards once per orbit, 
        so 2xHill sphere =4R_H worth of circular orbit will have possible encounter. 
        Thus, (4R_H/2pi a_circ1)= odds that a_circ1 is in the region of cross-over per orbit.
        For example, for BH at a_circ1 = 1e3r_g, 
            .. math:: R_h = a_{circ1}*(m_{circ1} + m_i/3M_{smbh})^1/3
            .. math:: = 0.004a_{circ1} (m_{circ1}/10M_{sun})^1/3 (m_i/10M_{sun})^1/3 (M_{smbh}/1e8M_{sun})^-1/3
        then
            ratio (4R_H/2pi a_circ1) = 0.008/pi ~ 0.0026 
            (ie around 1/400 odds that BH at a_circ1 is in either area of crossing)         

    6, calculate number of orbits of a_i in 1 timestep. 
        If e.g. N_orb(a_i)/timestep = 200 orbits per timestep of 10kyr, then 
        probability of encounter = (200orbits/timestep)*(4R_H/2pi a_circ1) ~ 0.5, 
                                or 50% odds of an encounter on this timestep between (a_circ1,j , a_i).
        If probability > 1, set probability = 1.
    7, draw a random number from the uniform [0,1] distribution and 
        if rng < probability of encounter, there is an encounter during the timestep
        if rng > probability of encounter, there is no encounter during the timestep

    8, if encounter:
        Take energy (de) from high ecc. a_i and give energy (de) to a_circ1,j
        de is average fractional energy change per encounter.
            So, a_circ1,j ->(1+de)a_circ1,j.    
                e_circ1,j ->(crit_ecc + de)
            and
                a_i       ->(1-de)a_i
                e_i       ->(1-de)e_i              
        Could be that average energy in gas-free cluster case is  
        assume average energy transfer = 20% perturbation (from Sigurdsson & Phinney 1993). 

        Further notes for self:
        sigma_ecc = sqrt(ecc^2 + incl^2)v_kep so if incl=0 deg (for now)
        En of ecc. interloper = 1/2 m_i sigma_ecc^2.
            Note: Can also use above logic for binary encounters except use binary binding energy instead.

        or later could try 
            Deflection angle defl = tan (defl) = dV_perp/V = 2GM/bV^2 kg^-1 m^3 s^-2 kg / m (m s^-1)^2
        so :math:`de/e =2GM/bV^2 = 2 G M_{bin}/0.5R_{hill}*\sigma^2`
        and :math:`R_hill = a_{circ1}*(M_{bin}/3M_{smbh})^1/3 and \sigma^2 =ecc^2*v_{kep}^2`
        So :math:`de/e = 4GM_{bin}/a_{circ1}(M_{bin}/3M_{smbh})^1/3 ecc^2 v_{kep}^2`
        and :math:`v_{kep} = \sqrt(GM_{smbh}/a_i)`
        So :math:`de/e = 4GM_{bin}^{2/3}M_{smbh}^1/3 a_i/a_{circ1} ecc^2 GM_{smbh} = 4(M_{bin}/M_{smbh})^{2/3} (a_i/a_{circ1})(1/ecc^2)
        where :math:`V_{rel} = \sigma` say and :math:`b=R_H = a_{circ1} (q/3)^{1/3}`
        So :math:`defl = 2GM/ a_{circ1}(q/3)^2/3 ecc^2 10^14 (m/s)^2 (R/10^3r_g)^-1`
            :math:`= 2 6.7e-11 2.e31/`
        !!Note: when doing this for binaries. 
            Calculate velocity of encounter compared to a_bin.
            If binary is hard ie GM_bin/a_bin > m3v_rel^2 then:
            harden binary 
                a_bin -> a_bin -da_bin and
            new binary eccentricity 
                e_bin -> e_bin + de  
            and give  da_bin worth of binding energy to extra eccentricity of m3.
            If binary is soft ie GM_bin/a_bin <m3v_rel^2 then:
            soften binary 
                a_bin -> a_bin + da_bin and
            new binary eccentricity
                e_bin -> e_bin + de
            and remove da_bin worth of binary energy from eccentricity of m3.
    """
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc <= disk_star_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc > disk_star_pro_orb_ecc_crit).nonzero()[0]

    if (len(circ_prograde_population_indices) == 0) or (len(ecc_prograde_population_indices) == 0):
        return disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, np.array([]), np.array([]), np.array([])

    # Put stellar radii in rg
    disk_star_pro_radius_rg = r_g_from_units(smbh_mass, ((10 ** disk_star_pro_radius) * u.Rsun)).value

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = (disk_radius_outer * ((disk_star_pro_masses[circ_prograde_population_indices] / (3 * (disk_star_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)))[:, None] * rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((disk_star_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    delta_energy_strong = np.exp(rng_here.normal(loc=np.log(delta_energy_strong_mu), scale=np.log(1. + delta_energy_strong_sigma), size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices))))
    num_poss_ints = 0
    num_encounters = 0
    id_nums_poss_touch = []
    frac_rhill_sep = []
    id_nums_unbound = []
    id_nums_flipped_rotation = []
    if len(circ_prograde_population_indices) > 0:
        for i, circ_idx in enumerate(circ_prograde_population_indices):
            for j, ecc_idx in enumerate(ecc_prograde_population_indices):
                if ((disk_star_pro_id_nums[ecc_idx] not in id_nums_flipped_rotation) and
                    (disk_star_pro_id_nums[circ_idx] not in id_nums_flipped_rotation) and
                    (disk_star_pro_id_nums[circ_idx] not in id_nums_unbound) and
                    (disk_star_pro_id_nums[ecc_idx] not in id_nums_unbound)):
                    if (disk_star_pro_orbs_a[circ_idx] < ecc_orb_max[j] and disk_star_pro_orbs_a[circ_idx] > ecc_orb_min[j]):
                        # prob_encounter/orbit =hill sphere size/circumference of circ orbit =2RH/2pi a_circ1
                        # r_h = a_circ1(temp_bin_mass/3smbh_mass)^1/3 so prob_enc/orb = mass_ratio^1/3/pi
                        temp_bin_mass = disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]
                        star_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                        mass_ratio_factor = (star_smbh_mass_ratio)**(1./3.)
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[i]
                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        if chance_of_enc[i][j] < prob_enc_per_timestep:
                            num_encounters = num_encounters + 1
                            # if close encounter, pump ecc of circ orbiter to e=0.1 from near circular, and incr a_circ1 by 10%
                            # drop ecc of a_i by 10% and drop a_i by 10% (P.E. = -GMm/a)
                            # if already pumped in eccentricity, no longer circular, so don't need to follow other interactions
                            if disk_star_pro_orbs_ecc[circ_idx] <= disk_star_pro_orb_ecc_crit:
                                new_orb_a_ecc, new_orb_a_circ, new_ecc_ecc, new_ecc_circ, id_num_out, id_num_flip = encounters_new_orba_ecc(
                                    smbh_mass,
                                    disk_star_pro_orbs_a[ecc_idx], disk_star_pro_orbs_a[circ_idx],
                                    disk_star_pro_masses[ecc_idx], disk_star_pro_masses[circ_idx],
                                    disk_star_pro_orbs_ecc[ecc_idx], disk_star_pro_orbs_ecc[circ_idx],
                                    disk_star_pro_radius_rg[ecc_idx], disk_star_pro_radius_rg[circ_idx],
                                    disk_star_pro_id_nums[ecc_idx], disk_star_pro_id_nums[circ_idx],
                                    delta_energy_strong[i][j], flag_obj_types=0, cardano = cardano)
                                if id_num_out is not None:
                                    id_nums_unbound.append(id_num_out)
                                if id_num_flip is not None:
                                    id_nums_flipped_rotation.append(id_num_flip)
                                # Check if any stars are outside the disk
                                if new_orb_a_ecc > disk_radius_outer:
                                    new_orb_a_ecc = disk_radius_outer - epsilon[i][j]
                                if new_orb_a_circ > disk_radius_outer:
                                    new_orb_a_circ = disk_radius_outer - epsilon[i][j]
                                disk_star_pro_orbs_a[ecc_idx] = new_orb_a_ecc
                                disk_star_pro_orbs_a[circ_idx] = new_orb_a_circ
                                disk_star_pro_orbs_ecc[circ_idx] = new_ecc_circ
                                disk_star_pro_orbs_ecc[ecc_idx] = new_ecc_ecc
                                # Look for stars that are inside each other's Hill spheres and if so return them as mergers
                                if (id_num_flip is None) and (id_num_out is None):
                                    separation = np.abs(disk_star_pro_orbs_a[circ_idx] - disk_star_pro_orbs_a[ecc_idx])
                                    center_of_mass = np.average([disk_star_pro_orbs_a[circ_idx], disk_star_pro_orbs_a[ecc_idx]],
                                                                weights=[disk_star_pro_masses[circ_idx], disk_star_pro_masses[ecc_idx]])
                                    rhill_poss_encounter = center_of_mass * ((disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]) / (3. * smbh_mass)) ** (1./3.)
                                    if (separation - rhill_poss_encounter < 0):
                                        id_nums_poss_touch.append(np.array([disk_star_pro_id_nums[circ_idx], disk_star_pro_id_nums[ecc_idx]]))
                                        frac_rhill_sep.append(separation / rhill_poss_encounter)

                        num_poss_ints = num_poss_ints + 1
            num_poss_ints = 0
            num_encounters = 0
    if not np.all(disk_star_pro_orbs_a > 0):
        zero_mask = ~(disk_star_pro_orbs_a > 0)
        print(disk_star_pro_orbs_a[zero_mask])
        print(np.argwhere(zero_mask))

    # Check finite
    assert np.isfinite(disk_star_pro_orbs_a).all(), \
        "Finite check failed for disk_star_pro_orbs_a"
    assert np.isfinite(disk_star_pro_orbs_ecc).all(), \
        "Finite check failed for disk_star_pro_orbs_ecc"
    assert np.all(disk_star_pro_orbs_a < disk_radius_outer), \
        "disk_star_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_star_pro_orbs_a > 0), \
        "disk_star_pro_orbs_a contains values <= 0"

    id_nums_poss_touch = np.array(id_nums_poss_touch)
    frac_rhill_sep = np.array(frac_rhill_sep)
    id_nums_unbound = np.array(id_nums_unbound)
    id_nums_flipped_rotation = np.array(id_nums_flipped_rotation)

    if id_nums_poss_touch.size > 0:
        # Check if any stars are marked as both unbound and within another star's Hill sphere
        # If yes, remove them from the within Hill sphere array
        if np.any(np.isin(id_nums_poss_touch, id_nums_unbound)):
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 0]) == True]
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 1]) == True]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 0]) == True, :]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 1]) == True, :]

        # Check if any stars are marked as both flipping from pro to retro and within another star's Hill sphere
        # If yes, remove them from the within Hill sphere array
        if np.any(np.isin(id_nums_flipped_rotation, id_nums_poss_touch)):
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 0]) == True]
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 1]) == True]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 0]) == True, :]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 1]) == True, :]

    # Test if there are any duplicate pairs, if so only return ID numbers of pair with smallest fractional Hill sphere separation
    if np.unique(id_nums_poss_touch).shape != id_nums_poss_touch.flatten().shape:
        sort_idx = np.argsort(frac_rhill_sep)
        id_nums_poss_touch = id_nums_poss_touch[sort_idx]
        uniq_vals, unq_counts = np.unique(id_nums_poss_touch, return_counts=True)
        dupe_vals = uniq_vals[unq_counts > 1]
        dupe_rows = id_nums_poss_touch[np.any(np.isin(id_nums_poss_touch, dupe_vals), axis=1)]
        uniq_rows = id_nums_poss_touch[np.all(~np.isin(id_nums_poss_touch, dupe_vals), axis=1)]

        rm_rows = []
        for row in dupe_rows:
            dupe_indices = np.any(np.isin(dupe_rows, row), axis=1).nonzero()[0][1:]
            rm_rows.append(dupe_indices)
        rm_rows = np.unique(np.concatenate(rm_rows))
        keep_mask = np.ones(len(dupe_rows))
        keep_mask[rm_rows] = 0

        id_nums_touch = np.concatenate((dupe_rows[keep_mask.astype(bool)], uniq_rows))

    else:
        id_nums_touch = id_nums_poss_touch

    id_nums_touch = id_nums_touch.T

    return (disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, id_nums_touch, id_nums_unbound, id_nums_flipped_rotation)

def circular_singles_encounters_prograde_stars_sweep(
        smbh_mass,
        disk_star_pro_orbs_a,
        disk_star_pro_masses,
        disk_star_pro_radius,
        disk_star_pro_orbs_ecc,
        disk_star_pro_id_nums,
        rstar_rhill_exponent,
        timestep_duration_yr,
        disk_star_pro_orb_ecc_crit,
        delta_energy_strong_mu,
        delta_energy_strong_sigma,
        disk_radius_outer,
        rng_here
        ):
    # Find the e< crit_ecc. population. These are the (circularized) population that can form binaries.
    circ_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc <= disk_star_pro_orb_ecc_crit).nonzero()[0]
    # Find the e> crit_ecc population. These are the interlopers that can perturb the circularized population
    ecc_prograde_population_indices = np.asarray(disk_star_pro_orbs_ecc > disk_star_pro_orb_ecc_crit).nonzero()[0]

    if (len(circ_prograde_population_indices) == 0) or (len(ecc_prograde_population_indices) == 0):
        return disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, np.array([]), np.array([]), np.array([])

    # Put stellar radii in rg
    disk_star_pro_radius_rg = r_g_from_units(smbh_mass, ((10 ** disk_star_pro_radius) * u.Rsun)).value

    # Calculate epsilon --amount to subtract from disk_radius_outer for objects with orb_a > disk_radius_outer
    epsilon = (disk_radius_outer * ((disk_star_pro_masses[circ_prograde_population_indices] / (3 * (disk_star_pro_masses[circ_prograde_population_indices] + smbh_mass)))**(1. / 3.)))[:, None] * rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))

    # T_orb = pi (R/r_g)^1.5 (GM_smbh/c^2) = pi (R/r_g)^1.5 (GM_smbh*2e30/c^2)
    #      = pi (R/r_g)^1.5 (6.7e-11 2e38/27e24)= pi (R/r_g)^1.5 (1.3e11)s =(R/r_g)^1/5 (1.3e4)
    orbital_timescales_circ_pops = scipy.constants.pi*((disk_star_pro_orbs_a[circ_prograde_population_indices])**(1.5))*(2.e30*smbh_mass*scipy.constants.G)/(scipy.constants.c**(3.0)*3.15e7) 
    N_circ_orbs_per_timestep = timestep_duration_yr/orbital_timescales_circ_pops
    ecc_orb_min = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0-disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    ecc_orb_max = disk_star_pro_orbs_a[ecc_prograde_population_indices]*(1.0+disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
    # Generate all possible needed random numbers ahead of time
    chance_of_enc = rng_here.uniform(size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices)))
    delta_energy_strong = np.exp(rng_here.normal(loc=np.log(delta_energy_strong_mu), scale=np.log(1. + delta_energy_strong_sigma), size=(len(circ_prograde_population_indices), len(ecc_prograde_population_indices))))
    num_poss_ints = 0
    num_encounters = 0
    id_nums_poss_touch = []
    frac_rhill_sep = []
    id_nums_unbound = []
    id_nums_flipped_rotation = []
    
    circ_len = len(circ_prograde_population_indices)
    ecc_len = len(ecc_prograde_population_indices)

    if  circ_len > 0:
        # if True engage the sweep algorithm

        # create the events array
        # define types to ensure correct sorting at boundary conditions:
        # START events are processed first, then POINTs, then ENDs
        START, POINT, END = -1, 0, 1
        
        # C = circ_prograde_population_indices.size
        # ecc_len = ecc_prograde_population_indices.size

        # create a single, flat, contiguous array for all events
        events = np.empty(circ_len + 2 * ecc_len, dtype=[('radius', 'f8'), ('type', 'i4'), ('rel_idx', 'u4')])

        # add POINT events for each circular object
        events[:circ_len] = np.array(list(zip(disk_star_pro_orbs_a[circ_prograde_population_indices], [POINT] * circ_len, np.arange(circ_len))), dtype=events.dtype)

        # Add START and ecc_lenND events for each eccentric object's interval
        ecc_orb_min = disk_star_pro_orbs_a[ecc_prograde_population_indices] * (1.0 - disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
        ecc_orb_max = disk_star_pro_orbs_a[ecc_prograde_population_indices] * (1.0 + disk_star_pro_orbs_ecc[ecc_prograde_population_indices])
        events[circ_len:circ_len+ecc_len] = np.array(list(zip(ecc_orb_min, [START] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)
        events[circ_len+ecc_len:] = np.array(list(zip(ecc_orb_max, [END] * ecc_len, np.arange(ecc_len))), dtype=events.dtype)

        # sort the events by radius
        # uses numpy sort, very performant
        events.sort(order=['radius', 'type'])

        # for a two-pass system to ensure parity, we collect
        # the overlaps up front
        overlaps = {}

        # first we sweep in order to construct the overlaps object
        active_ecc_indices = set()

        for radius, type, rel_idx in events:
            if type == START:
                active_ecc_indices.add(rel_idx)
            elif type == END:
                active_ecc_indices.discard(rel_idx) # Use discard for safety
            elif type == POINT:
                # when we hit a POINT event, the `active_ecc_indices` set contains
                # ALL eccentric particles whose intervals contain this point
                if not active_ecc_indices:
                    continue

                circ_rel_idx = rel_idx

                if active_ecc_indices: 
                    sorted_interlopers = sorted(list(active_ecc_indices))
                    overlaps[circ_rel_idx] = sorted_interlopers

        # turn these lists into sets for the time being in order to much more effectively 
        # search them for the indices
        # we will turn them back into arrays later for return
        id_nums_flipped_rotation = set(id_nums_flipped_rotation)
        id_nums_unbound = set(id_nums_unbound)

        # now we actualy go through the dictionary of interlopers
        for circ_rel_idx in range(len(circ_prograde_population_indices)):
            circ_idx = circ_prograde_population_indices[circ_rel_idx]
                
            # interlopers = overlaps.get(circ_rel_idx, [])
            interlopers_set = set(overlaps.get(circ_rel_idx, []))

            if not interlopers_set:
                continue

            for ecc_rel_idx in range(len(ecc_prograde_population_indices)):
                if ecc_rel_idx in interlopers_set:
                    ecc_idx = ecc_prograde_population_indices[ecc_rel_idx]
            # for ecc_rel_idx in interlopers:
            #     ecc_idx = ecc_prograde_population_indices[ecc_rel_idx]

                    if ((disk_star_pro_id_nums[ecc_idx] not in id_nums_flipped_rotation) and
                        (disk_star_pro_id_nums[circ_idx] not in id_nums_flipped_rotation) and
                        (disk_star_pro_id_nums[circ_idx] not in id_nums_unbound) and
                        (disk_star_pro_id_nums[ecc_idx] not in id_nums_unbound)):

                        temp_bin_mass = disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]
                        star_smbh_mass_ratio = temp_bin_mass/(3.0*smbh_mass)
                        mass_ratio_factor = (star_smbh_mass_ratio)**(1./3.)
                        prob_orbit_overlap = (1./scipy.constants.pi)*mass_ratio_factor
                        prob_enc_per_timestep = prob_orbit_overlap * N_circ_orbs_per_timestep[circ_rel_idx]
                        if prob_enc_per_timestep > 1:
                            prob_enc_per_timestep = 1
                        if chance_of_enc[circ_rel_idx][ecc_rel_idx] < prob_enc_per_timestep:
                            if disk_star_pro_orbs_ecc[circ_idx] <= disk_star_pro_orb_ecc_crit:
                                new_orb_a_ecc, new_orb_a_circ, new_ecc_ecc, new_ecc_circ, id_num_out, id_num_flip = encounters_new_orba_ecc(
                                    smbh_mass,
                                    disk_star_pro_orbs_a[ecc_idx], disk_star_pro_orbs_a[circ_idx],
                                    disk_star_pro_masses[ecc_idx], disk_star_pro_masses[circ_idx],
                                    disk_star_pro_orbs_ecc[ecc_idx], disk_star_pro_orbs_ecc[circ_idx],
                                    disk_star_pro_radius_rg[ecc_idx], disk_star_pro_radius_rg[circ_idx],
                                    disk_star_pro_id_nums[ecc_idx], disk_star_pro_id_nums[circ_idx],
                                    delta_energy_strong[circ_rel_idx][ecc_rel_idx], flag_obj_types=0)
                                if id_num_out is not None:
                                    id_nums_unbound.add(id_num_out)
                                if id_num_flip is not None:
                                    id_nums_flipped_rotation.add(id_num_flip)
                                # Check if any stars are outside the disk
                                if new_orb_a_ecc > disk_radius_outer:
                                    new_orb_a_ecc = disk_radius_outer - epsilon[circ_rel_idx][ecc_rel_idx]
                                if new_orb_a_circ > disk_radius_outer:
                                    new_orb_a_circ = disk_radius_outer - epsilon[circ_rel_idx][ecc_rel_idx]
                                disk_star_pro_orbs_a[ecc_idx] = new_orb_a_ecc
                                disk_star_pro_orbs_a[circ_idx] = new_orb_a_circ
                                disk_star_pro_orbs_ecc[circ_idx] = new_ecc_circ
                                disk_star_pro_orbs_ecc[ecc_idx] = new_ecc_ecc
    #                             # Look for stars that are inside each other's Hill spheres and if so return them as mergers
                                if (id_num_flip is None) and (id_num_out is None):
                                    separation = np.abs(disk_star_pro_orbs_a[circ_idx] - disk_star_pro_orbs_a[ecc_idx])
                                    center_of_mass = np.average([disk_star_pro_orbs_a[circ_idx], disk_star_pro_orbs_a[ecc_idx]],
                                                                weights=[disk_star_pro_masses[circ_idx], disk_star_pro_masses[ecc_idx]])
                                    rhill_poss_encounter = center_of_mass * ((disk_star_pro_masses[circ_idx] + disk_star_pro_masses[ecc_idx]) / (3. * smbh_mass)) ** (1./3.)
                                    if (separation - rhill_poss_encounter < 0):
                                        id_nums_poss_touch.append(np.array([disk_star_pro_id_nums[circ_idx], disk_star_pro_id_nums[ecc_idx]]))
                                        frac_rhill_sep.append(separation / rhill_poss_encounter)
    if not np.all(disk_star_pro_orbs_a > 0):
        zero_mask = ~(disk_star_pro_orbs_a > 0)
        print(disk_star_pro_orbs_a[zero_mask])
        print(np.argwhere(zero_mask))

    # Check finite
    assert np.isfinite(disk_star_pro_orbs_a).all(), \
        "Finite check failed for disk_star_pro_orbs_a"
    assert np.isfinite(disk_star_pro_orbs_ecc).all(), \
        "Finite check failed for disk_star_pro_orbs_ecc"
    assert np.all(disk_star_pro_orbs_a < disk_radius_outer), \
        "disk_star_pro_orbs_a contains values greater than disk_radius_outer"
    assert np.all(disk_star_pro_orbs_a > 0), \
        "disk_star_pro_orbs_a contains values <= 0"

    id_nums_poss_touch = np.array(id_nums_poss_touch)
    frac_rhill_sep = np.array(frac_rhill_sep)
    id_nums_unbound = np.array(id_nums_unbound)
    id_nums_flipped_rotation = np.array(id_nums_flipped_rotation)

    if id_nums_poss_touch.size > 0:
        # Check if any stars are marked as both unbound and within another star's Hill sphere
        # If yes, remove them from the within Hill sphere array
        if np.any(np.isin(id_nums_poss_touch, id_nums_unbound)):
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 0]) == True]
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 1]) == True]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 0]) == True, :]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_unbound)[:, 1]) == True, :]

        # Check if any stars are marked as both flipping from pro to retro and within another star's Hill sphere
        # If yes, remove them from the within Hill sphere array
        if np.any(np.isin(id_nums_flipped_rotation, id_nums_poss_touch)):
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 0]) == True]
            frac_rhill_sep = frac_rhill_sep[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 1]) == True]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 0]) == True, :]
            id_nums_poss_touch = id_nums_poss_touch[~(np.isin(id_nums_poss_touch, id_nums_flipped_rotation)[:, 1]) == True, :]

    # Test if there are any duplicate pairs, if so only return ID numbers of pair with smallest fractional Hill sphere separation
    if np.unique(id_nums_poss_touch).shape != id_nums_poss_touch.flatten().shape:
        sort_idx = np.argsort(frac_rhill_sep)
        id_nums_poss_touch = id_nums_poss_touch[sort_idx]
        uniq_vals, unq_counts = np.unique(id_nums_poss_touch, return_counts=True)
        dupe_vals = uniq_vals[unq_counts > 1]
        dupe_rows = id_nums_poss_touch[np.any(np.isin(id_nums_poss_touch, dupe_vals), axis=1)]
        uniq_rows = id_nums_poss_touch[np.all(~np.isin(id_nums_poss_touch, dupe_vals), axis=1)]

        rm_rows = []
        for row in dupe_rows:
            dupe_indices = np.any(np.isin(dupe_rows, row), axis=1).nonzero()[0][1:]
            rm_rows.append(dupe_indices)
        rm_rows = np.unique(np.concatenate(rm_rows))
        keep_mask = np.ones(len(dupe_rows))
        keep_mask[rm_rows] = 0

        id_nums_touch = np.concatenate((dupe_rows[keep_mask.astype(bool)], uniq_rows))

    else:
        id_nums_touch = id_nums_poss_touch

    id_nums_touch = id_nums_touch.T

    return (disk_star_pro_orbs_a, disk_star_pro_orbs_ecc, id_nums_touch, id_nums_unbound, id_nums_flipped_rotation)




def generate_data_stars(N: int, circ_proportion: float, singleton_proportion: float, rng: np.random.Generator):
    """Generates random mock data for the star simulation functions."""
    # Mock physical constants
    mock_params = {
        "smbh_mass": 1.0e8,
        "timestep_duration_yr": 1.0e4,
        "disk_star_pro_orb_ecc_crit": 0.1,
        "rstar_rhill_exponent": 2, # 2 is the default
        "delta_energy_strong_mu": 0.11,
        "delta_energy_strong_sigma": 0.05,
        "disk_radius_outer": 20000.0
    }

    # Generate random BH properties
    num_circ = int(N * circ_proportion)
    num_ecc = N - num_circ

    # Generate eccentricities to match the desired C/E proportion
    ecc_crit = mock_params["disk_star_pro_orb_ecc_crit"]
    e_circ = rng.uniform(0, ecc_crit, size=num_circ)
    e_ecc = rng.uniform(ecc_crit * 1.01, 0.8, size=num_ecc) # Ensure e > e_crit
    
    # Combine and shuffle to avoid any ordering bias
    all_eccs = np.concatenate((e_circ, e_ecc))
    rng.shuffle(all_eccs)
    
    mock_params["disk_star_pro_orbs_ecc"] = all_eccs
    mock_params["disk_star_pro_masses"] = rng.uniform(5, 50, size=N)
    # Ensure semi-major axis is always positive and within the disk
    mock_params["disk_star_pro_orbs_a"] = rng.uniform(
        100, mock_params["disk_radius_outer"] * 0.95, size=N
    )
    mock_params["disk_star_pro_radius"] = rng.uniform(1, 20, size = N)
    # mocking a certain proportion of these stars as singletons, non-repeating
    mock_params["disk_star_pro_id_nums"] = np.arange(0, N)
    
    return mock_params

def run_benchmark_stars(N: int, circ_proportion: float):
    """Runs a single benchmark test for a given N and C/E proportion."""
    print(f"--- Testing: N={N}, C/E Proportion={circ_proportion:.2f} ---")

    data_rng = np.random.default_rng(seed=123)
    data = generate_data_stars(N, circ_proportion, singleton_proportion = SINGLETON_PROP, rng = data_rng)

    # We need separate, identically-seeded RNGs for the functions themselves
    # to ensure they use the same random numbers internally for the test.
    rng1 = np.random.default_rng(seed=456)
    rng2 = np.random.default_rng(seed=456)
    

    # --- Run Original Function ---
    # Important: Copy data as the functions modify arrays in-place
    data_for_orig = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_orig, ecc_orig, id_nums_touch_orig, id_nums_unbound_orig, id_nums_flipped_rotation_orig = circular_singles_encounters_prograde_stars(**data_for_orig, rng_here=rng1, cardano=False)
    time_orig = time.perf_counter() - start_time
    print(f"Original took:   {time_orig:.4f} seconds")

    # --- Run Optimized Function ---
    data_for_opt = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    start_time = time.perf_counter()
    a_opt, ecc_opt, id_nums_touch_opt, id_nums_unbound_opt, id_nums_flipped_rotation_opt = circular_singles_encounters_prograde_stars(**data_for_opt, rng_here=rng2, cardano = True)
    time_opt = time.perf_counter() - start_time
    print(f"Optimized took:  {time_opt:.4f} seconds")

    # --- Correctness ---
    correct_a = np.allclose(a_orig, a_opt, rtol = AXIS_TOLERANCE)
    correct_ecc = np.allclose(ecc_orig, ecc_opt, rtol = ECCENTRICITY_TOLERANCE) # the eccentricities are much less reliable than the axes
    correct_touch = np.all(id_nums_touch_orig == id_nums_touch_opt)
    correct_unbound = np.all(id_nums_unbound_orig == id_nums_unbound_opt)
    correct_flipped_rotation = np.all(id_nums_flipped_rotation_orig == id_nums_flipped_rotation_opt)

    assert correct_a, \
        "The returned semi-major axes were not within specified tolerances"

    assert correct_ecc, \
        "The returned eccentricities were not within specified tolerances"

    assert correct_touch, \
        "The returned touch ids were not identical"

    assert correct_unbound, \
        "The returned unbound ids were not identical"

    assert correct_flipped_rotation, \
        "The returned flipped rotation ids were not identical"


    speedup = time_orig / time_opt if time_opt > 0 else float('inf')
    print(f"🚀 Speedup: {speedup:.2f}x")

    # --- Speedup ---
    # we should at least see parity, and never see a considerable slowdown
    # otherwise, we haven't set the length check correctly and we're using an ill-suited algorithm

    # speedup = time_orig / time_opt if time_opt > 0 else float('inf')
    # assert speedup > SPEED_TOLERANCE, \
    #     "We see a considerable slowdown"



if __name__ == "__main__":
    # Define the set of test cases to run
    test_cases = [
        # Test scaling with N at a 50/50 C/E split
        (10, 0.5),
        (20, 0.5),
        (30, 0.5),
        (40, 0.5),
        (50, 0.5),
        (100, 0.5),
        (250, 0.5),   
        (500, 0.5),
        (1000, 0.5),
        (2000, 0.5),
        # (4000, 0.5),  
        # (8000, 0.5),
        # (10000, 0.5),  
        # (15000, 0.5),
        # (20000, 0.5),
        # (25000, 0.5),
        # (30000, 0.5),
        # (40000, 0.5),
        # (60000, 0.5),
        # (80000, 0.5),
        # (100000, 0.5),
        # (160000, 0.5),
        # The C*E product is largest at 0.5, so these should be faster for the original
    ]

    for N, prop in test_cases:
        run_benchmark_stars(N, prop)
