#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinematics of a steerable five-link axle (optionally with bell crank).

Structure of this file:
    BLOCK 0      : input data of the simulation
    BLOCK 1 - 7  : Kinematics                     - multibody kinematics of the axle
    BLOCK 8 - 16 : LiveMovementBase               - sweeps, characteristics and plots
    BLOCK 17     : LiveMovementStd5Link           - translational steering input
    BLOCK 18     : LiveMovementRotationAxis5Link  - rotational steering input
    BLOCK 19     : main program

Notation follows the documentation:
    body a : chassis side  (chassis hardpoints / inner knuckle)
    body b : wheel carrier (outer knuckle)
    body d : bell crank
    beta   : primary joint coordinates [phi, theta, psi, x, y]   (z is prescribed)
    eta    : velocity level unknowns   [omega_bx, omega_by, omega_bz, v_bx, v_by]
    g^s    : implicit closure conditions, G^s_a / G^s_b the corresponding
             constraint matrices of body a and body b

All lengths are in mm, all angles in rad unless a plot label says otherwise.

@author: eric
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import Slider, Button
import time


# =============================================================================
# BLOCK 0: INPUT DATA OF THE SIMULATION
# =============================================================================

# axle type that is simulated: "std" = conventional five-link axle with a
# translational steering input, "rot" = five-link axle whose chassis hardpoints
# rotate about the axis of the inner knuckle
axle_config = "rot"


# --- ride height sweep, identical for both axle types ------------------------

z_min = -120          # lower end of the ride height range
z_max = 100           # upper end of the ride height range
z_step = 5            # step width of the coarse start value grid

zDot = 1000           # prescribed vertical velocity of the wheel carrier
z2Dot = 0             # prescribed vertical acceleration of the wheel carrier


# --- tire 215/60R18 ----------------------------------------------------------

tire_radius = 358
tire_width = 215


# --- loads -------------------------------------------------------------------

F_w = np.array([-10550, 0, 10550])   # force at the wheel contact patch
F_t = np.array([0, 0, 0])          # additional force at the wheel center


# =============================================================================
# STD AXLE: conventional five-link axle, translational steering input
# =============================================================================

# --- steering sweep, the steering input is a translation ---------------------

steer_min_STD = -0
steer_max_STD = 0
steer_step_STD = 2.5

steerDot_STD = 100    # prescribed steering velocity
steer2Dot_STD = 0     # prescribed steering acceleration


# --- chassis hardpoints in design position (P0) ------------------------------

P_a0P0_STD = np.array([-250, 315, -90])
P_a1P0_STD = np.array([280, 315, -90])
P_a2P0_STD = np.array([-180, 315, -90])
P_a3P0_STD = np.array([-180, 315, 390])
P_a4P0_STD = np.array([280, 315, 390])
P_a5P0_STD = np.array([60, 328.88, 350])   # mounting point of the spring/damper element
P_aP0_STD = [P_a0P0_STD, P_a1P0_STD, P_a2P0_STD, P_a3P0_STD, P_a4P0_STD, P_a5P0_STD]


# --- wheel carrier hardpoints in design position (P0) ------------------------

r_bP0_STD = np.array([100, 850, 0])        # wheel center, the z value MUST be 0!
P_b0P0_STD = np.array([-150, 720, -115])
P_b1P0_STD = np.array([100, 720, -115])
P_b2P0_STD = np.array([100, 720, -115])
P_b3P0_STD = np.array([100, 720, 425])
P_b4P0_STD = np.array([100, 720, 425])
P_b5P0_STD = np.array([100, 720, 425])     # spring/damper resp. push-/pullrod
# point on the wheel rotation axis, its x and z value set the static toe and camber
P_b6P0_STD = np.array([0, -60, 0]) + r_bP0_STD
P_bP0_STD = [P_b0P0_STD, P_b1P0_STD, P_b2P0_STD, P_b3P0_STD, P_b4P0_STD, P_b5P0_STD, P_b6P0_STD]


# --- steered links and link diameters ----------------------------------------

# selection vector, marks the links that follow the steering input (here link 0)
a_i_STD = [1, 0, 0, 0, 0]

# diameters of the five links and of the spring/damper resp. push-/pullrod, only used for the 3d view
link_diameters_STD = [25.0, 25.0, 25.0, 25.0, 25.0, 0]


# --- bell crank and spring/damper element ------------------------------------

P_d0P0_STD = np.array([100, 0, 350])       # joint of the push-/pullrod
P_d1P0_STD = np.array([100, -150, 200])    # joint of the spring/damper element
P_dP0_STD = [P_d0P0_STD, P_d1P0_STD]

r_dP0_STD = np.array([100, -50, 200])      # pivot of the bell crank

# rotation axis of the bell crank, perpendicular to the plane of its three points
u_d_unscaled_STD = np.cross(P_d1P0_STD - P_d0P0_STD, r_dP0_STD - P_d0P0_STD)
u_d_STD = u_d_unscaled_STD / np.linalg.norm(u_d_unscaled_STD)

# True: spring/damper element driven by a bell crank, False: directly mounted
has_bell_crank_STD = False


# =============================================================================
# ROT AXLE: five-link axle with a rotating inner knuckle
# =============================================================================

# --- steering sweep, the steering input is an angle in rad -------------------

steer_min_ROT = -0.85
steer_max_ROT = 0.30
steer_step_ROT = 0.01

steerDot_ROT = 3      # prescribed steering velocity
steer2Dot_ROT = 3     # prescribed steering acceleration


# --- chassis hardpoints in design position (P0) ------------------------------

P_a0P0_ROT = np.array([-420, 419, 23])     # P1
P_a1P0_ROT = np.array([240, 299, -87])     # P3
P_a2P0_ROT = np.array([-120, 328, -87])    # P5
P_a3P0_ROT = np.array([-120, 300, 390])    # P7
P_a4P0_ROT = np.array([228, 271, 390])     # P9
P_a5P0_ROT = np.array([60, 450, 600])      # P11, mounting point of the spring/damper
P_aP0_ROT = [P_a0P0_ROT, P_a1P0_ROT, P_a2P0_ROT, P_a3P0_ROT, P_a4P0_ROT, P_a5P0_ROT]

# origin of the body fixed frames K_a, identical for all five chassis hardpoints
# because they all rotate about the same axis of the inner knuckle
r_a_ROT = np.array([60, 335, 72])
r_aP0_ROT = np.tile(r_a_ROT, (5, 1))


# --- wheel carrier hardpoints in design position (P0) ------------------------

# wheel center, the z value MUST be 0! Slight offset because of the pneumatic trail
r_bP0_ROT = np.array([112, 850, 0])
P_b0P0_ROT = np.array([138, 713, 0])       # P2
P_b1P0_ROT = np.array([240, 725, -115])    # P4
P_b2P0_ROT = np.array([30, 707, -115])     # P6
P_b3P0_ROT = np.array([24, 691, 424])      # P8
P_b4P0_ROT = np.array([228, 709, 424])     # P10
P_b5P0_ROT = np.array([140, 700, 90])      # P12, spring/damper resp. push-/pullrod
# point on the wheel rotation axis, its x and z value set the static toe and camber
P_b6P0_ROT = np.array([0, -60, -0.54]) + r_bP0_ROT
P_bP0_ROT = [P_b0P0_ROT, P_b1P0_ROT, P_b2P0_ROT, P_b3P0_ROT, P_b4P0_ROT, P_b5P0_ROT, P_b6P0_ROT]


# --- steered links and rotation axis of the inner knuckle --------------------

# selection vector, marks the links that follow the steering input (here links 1 to 4)
a_i_ROT = [0, 1, 1, 1, 1]

# rotation axis of the inner knuckle, identical for all five chassis hardpoints
u_a_unscaled_ROT = np.array([0, -0.04, 1])
u_a_hat_ROT = u_a_unscaled_ROT / np.linalg.norm(u_a_unscaled_ROT)
u_a_ROT = np.tile(u_a_hat_ROT, (5, 1))

# bearings on the rotation axis of the inner knuckle, only used for the 3d view
bearing_sizes_ROT = [
    [-160, -240, 100],  # bearing 1: z_start, z_end, diameter
    [480, 560, 100]     # bearing 2: z_start, z_end, diameter
    ]

# diameters of the five links and of the spring/damper element resp. push-/pullrod, only used for the 3d view
link_diameters_ROT = [35.0, 35.0, 35.0, 35.0, 35.0, 60]


# --- bell crank and spring/damper element ------------------------------------

P_d0P0_ROT = np.array([15, 300, 700])      # joint of the push-/pullrod
P_d1P0_ROT = np.array([160, 200, 820])     # joint of the spring/damper element
P_dP0_ROT = [P_d0P0_ROT, P_d1P0_ROT]

r_dP0_ROT = np.array([175, 270, 740])      # pivot of the bell crank

# rotation axis of the bell crank, perpendicular to the plane of its three points
u_d_unscaled_ROT = np.cross(P_d1P0_ROT - P_d0P0_ROT, r_dP0_ROT - P_d0P0_ROT)
u_d_ROT = u_d_unscaled_ROT / np.linalg.norm(u_d_unscaled_ROT)

# True: spring/damper element driven by a bell crank, False: directly mounted
has_bell_crank_ROT = False


# # =============================================================================
# # ROT AXLE: five-link axle with a rotating inner knuckle reduced to 2d problem
# # =============================================================================

# # --- steering sweep, the steering input is an angle in rad -------------------

# steer_min_ROT = -0.85
# steer_max_ROT = 0.30
# steer_step_ROT = 0.01

# steerDot_ROT = 3      # prescribed steering velocity
# steer2Dot_ROT = 3     # prescribed steering acceleration


# # --- chassis hardpoints in design position (P0) ------------------------------

# P_a0P0_ROT = np.array([-420, 425, 0])      # P1
# P_a1P0_ROT = np.array([240, 300, -200])    # P3
# P_a2P0_ROT = np.array([-120, 330, -200])   # P5
# P_a3P0_ROT = np.array([-120, 330, 200])    # P7
# P_a4P0_ROT = np.array([240, 300, 200])     # P9
# P_a5P0_ROT = np.array([0, 0, 0])           # P11, mounting point of the spring/damper
# P_aP0_ROT = [P_a0P0_ROT, P_a1P0_ROT, P_a2P0_ROT, P_a3P0_ROT, P_a4P0_ROT, P_a5P0_ROT]

# # origin of the body fixed frames K_a, identical for all five chassis hardpoints
# # because they all rotate about the same axis of the inner knuckle
# r_a_ROT = np.array([60, 340, 0])
# r_aP0_ROT = np.tile(r_a_ROT, (5, 1))


# # --- wheel carrier hardpoints in design position (P0) ------------------------

# # wheel center, the z value MUST be 0! Slight offset because of the pneumatic trail
# r_bP0_ROT = np.array([110, 850, 0])
# P_b0P0_ROT = np.array([140, 720, 0])       # P2
# P_b1P0_ROT = np.array([240, 730, -200])    # P4
# P_b2P0_ROT = np.array([30, 710, -200])     # P6
# P_b3P0_ROT = np.array([30, 710, 200])      # P8
# P_b4P0_ROT = np.array([240, 730, 200])     # P10
# P_b5P0_ROT = np.array([240, 730, 200])     # P12, spring/damper resp. push-/pullrod
# # point on the wheel rotation axis, its x and z value set the static toe and camber
# P_b6P0_ROT = np.array([0, -60, 0]) + r_bP0_ROT
# P_bP0_ROT = [P_b0P0_ROT, P_b1P0_ROT, P_b2P0_ROT, P_b3P0_ROT, P_b4P0_ROT, P_b5P0_ROT, P_b6P0_ROT]


# # --- steered links and rotation axis of the inner knuckle --------------------

# # selection vector, marks the links that follow the steering input (here links 1 to 4)
# a_i_ROT = [0, 1, 1, 1, 1]

# # rotation axis of the inner knuckle, identical for all five chassis hardpoints
# u_a_unscaled_ROT = np.array([0, 0, 1])
# u_a_hat_ROT = u_a_unscaled_ROT / np.linalg.norm(u_a_unscaled_ROT)
# u_a_ROT = np.tile(u_a_hat_ROT, (5, 1))

# # bearings on the rotation axis of the inner knuckle, only used for the 3d view
# bearing_sizes_ROT = [
#     [-220, -320, 100],  # bearing 1: z_start, z_end, diameter
#     [220, 320, 100]     # bearing 2: z_start, z_end, diameter
#     ]

# # diameters of the five links and of the spring/damper element resp. push-/pullrod, only used for the 3d view
# link_diameters_ROT = [25.0, 35.0, 35.0, 25.0, 25.0, 0]


# # --- bell crank and spring/damper element ------------------------------------

# P_d0P0_ROT = np.array([15, 300, 700])      # joint of the push-/pullrod
# P_d1P0_ROT = np.array([160, 200, 820])     # joint of the spring/damper element
# P_dP0_ROT = [P_d0P0_ROT, P_d1P0_ROT]

# r_dP0_ROT = np.array([175, 270, 740])      # pivot of the bell crank

# # rotation axis of the bell crank, perpendicular to the plane of its three points
# u_d_unscaled_ROT = np.cross(P_d1P0_ROT - P_d0P0_ROT, r_dP0_ROT - P_d0P0_ROT)
# u_d_ROT = u_d_unscaled_ROT / np.linalg.norm(u_d_unscaled_ROT)

# # True: spring/damper element driven by a bell crank, False: directly mounted
# has_bell_crank_ROT = False


# # =============================================================================
# # ROT AXLE: five-link axle with a rotating inner knuckle baseline 3d
# # =============================================================================

# # --- steering sweep, the steering input is an angle in rad -------------------

# steer_min_ROT = -0.85
# steer_max_ROT = 0.30
# steer_step_ROT = 0.01

# steerDot_ROT = 3      # prescribed steering velocity
# steer2Dot_ROT = 3     # prescribed steering acceleration


# # --- chassis hardpoints in design position (P0) ------------------------------

# P_a0P0_ROT = np.array([-420, 425, 0])      # P1
# P_a1P0_ROT = np.array([240, 300, -90])     # P3
# P_a2P0_ROT = np.array([-120, 330, -90])    # P5
# P_a3P0_ROT = np.array([-120, 330, 390])    # P7
# P_a4P0_ROT = np.array([240, 300, 390])     # P9
# P_a5P0_ROT = np.array([60, 450, 600])      # P11, mounting point of the spring/damper
# P_aP0_ROT = [P_a0P0_ROT, P_a1P0_ROT, P_a2P0_ROT, P_a3P0_ROT, P_a4P0_ROT, P_a5P0_ROT]

# # origin of the body fixed frames K_a, identical for all five chassis hardpoints
# # because they all rotate about the same axis of the inner knuckle
# r_a_ROT = np.array([60, 340, 0])
# r_aP0_ROT = np.tile(r_a_ROT, (5, 1))


# # --- wheel carrier hardpoints in design position (P0) ------------------------

# # wheel center, the z value MUST be 0! Slight offset because of the pneumatic trail
# r_bP0_ROT = np.array([110, 850, 0])
# P_b0P0_ROT = np.array([140, 720, 0])       # P2
# P_b1P0_ROT = np.array([240, 730, -115])    # P4
# P_b2P0_ROT = np.array([30, 710, -115])     # P6
# P_b3P0_ROT = np.array([30, 710, 425])      # P8
# P_b4P0_ROT = np.array([240, 730, 425])     # P10
# P_b5P0_ROT = np.array([140, 700, 90])      # P12, spring/damper resp. push-/pullrod
# # point on the wheel rotation axis, its x and z value set the static toe and camber
# P_b6P0_ROT = np.array([0, -60, 0]) + r_bP0_ROT
# P_bP0_ROT = [P_b0P0_ROT, P_b1P0_ROT, P_b2P0_ROT, P_b3P0_ROT, P_b4P0_ROT, P_b5P0_ROT, P_b6P0_ROT]


# # --- steered links and rotation axis of the inner knuckle --------------------

# # selection vector, marks the links that follow the steering input (here links 1 to 4)
# a_i_ROT = [0, 1, 1, 1, 1]

# # rotation axis of the inner knuckle, identical for all five chassis hardpoints
# u_a_unscaled_ROT = np.array([0, 0, 1])
# u_a_hat_ROT = u_a_unscaled_ROT / np.linalg.norm(u_a_unscaled_ROT)
# u_a_ROT = np.tile(u_a_hat_ROT, (5, 1))

# # bearings on the rotation axis of the inner knuckle, only used for the 3d view
# bearing_sizes_ROT = [
#     [-160, -240, 100],  # bearing 1: z_start, z_end, diameter
#     [480, 560, 100]     # bearing 2: z_start, z_end, diameter
#     ]

# # diameters of the five links and of the spring/damper element resp. push-/pullrod, only used for the 3d view
# link_diameters_ROT = [25.0, 35.0, 35.0, 25.0, 25.0, 60]


# # --- bell crank and spring/damper element ------------------------------------

# P_d0P0_ROT = np.array([15, 300, 700])      # joint of the push-/pullrod
# P_d1P0_ROT = np.array([160, 200, 820])     # joint of the spring/damper element
# P_dP0_ROT = [P_d0P0_ROT, P_d1P0_ROT]

# r_dP0_ROT = np.array([175, 270, 740])      # pivot of the bell crank

# # rotation axis of the bell crank, perpendicular to the plane of its three points
# u_d_unscaled_ROT = np.cross(P_d1P0_ROT - P_d0P0_ROT, r_dP0_ROT - P_d0P0_ROT)
# u_d_ROT = u_d_unscaled_ROT / np.linalg.norm(u_d_unscaled_ROT)

# # True: spring/damper element driven by a bell crank, False: directly mounted
# has_bell_crank_ROT = False



class Kinematics:
    """
    Multibody kinematics of the five-link axle.

    Index convention of the hardpoint arrays:
        P_a[0:5] / P_b[0:5] : the five links
        P_a[5]              : chassis side mounting point of the spring/damper element
        P_b[5]              : wheel carrier mounting point of the spring/damper element
                              resp. of the push-/pullrod (bell crank configuration)
        P_b[6]              : point on the wheel rotation axis (defines camber and toe)
        P_d[0] / P_d[1]     : bell crank hardpoints of push-/pullrod and spring/damper

    Typical call order for one suspension position:
        set_geometry() -> solve_position() -> calc_position()
        [-> solve_bell_crank() -> calc_bell_crank()]
        -> calc_Gs_a() -> calc_Gs_b() -> velocity level / characteristics / forces
    (calc_geometry() performs this sequence in one call.)
    """

    # Right hand side variants of the closure conditions on velocity level
    MODE_FULL = 0    # heave and steering superimposed
    MODE_HEAVE = 1   # pure vertical motion (bump / rebound), steering fixed
    MODE_STEER = 2   # pure steering motion, ride height fixed

    # Tolerance for degeneracy checks (pure translation, singular directions)
    EPS = 1e-12

    def __init__(self):
        # --- geometry, defined by set_geometry() -----------------------------
        self.a_c_a = None            # {}^a c_a : body fixed lever arms, chassis side (5, 3)
        self.b_c_b = None            # {}^b c_b : body fixed lever arms, wheel carrier (7, 3)
        self.d_c_d = None            # {}^d c_d : body fixed lever arms, bell crank (2, 3)
        self.l_i_const = None        # constant link lengths l of the five links (5,)
        self.l_d_const = None        # constant length of the push-/pullrod
        self.c_a = None              # c_a in the inertial frame K_0 (5, 3)
        self.P_a = None              # chassis side hardpoints (6, 3)
        self.r_d = None              # position of the bell crank pivot

        # --- position level --------------------------------------------------
        self.beta = None             # primary joint coordinates [phi, theta, psi, x, y]
        self.r_b = None              # origin of the wheel carrier frame K_b = [x, y, z]
        self.c_b = None              # c_b in the inertial frame K_0 (7, 3)
        self.P_b = None              # wheel carrier hardpoints (7, 3)
        self.l_i = None              # distance vectors l_i = P_b,i - P_a,i (5, 3)
        self.r_wb = None             # vector from wheel center to wheel contact patch
        self.phi_d = None            # primary joint coordinate of the bell crank
        self.c_d = None              # c_d in the inertial frame K_0 (2, 3)
        self.P_d = None              # bell crank hardpoints (2, 3)
        self.l_d0 = None             # push-/pullrod, P_d,0 - P_b,5
        self.l_d1 = None             # spring/damper element of the bell crank, P_d,1 - P_a,5
        self.l_5 = None              # spring/damper resp. push-/pullrod, P_b,5 - P_a,5
        self.spring_damper_length = None
        self.P_axis_a_upper = None   # display points of the inner knuckle rotation axis
        self.P_axis_a_lower = None

        # --- velocity / acceleration level -----------------------------------
        self.Gs_a = None             # constraint matrix G^s_a (5, 6)
        self.Gs_b_A = None           # part of G^s_b belonging to the unknowns (5, 5)
        self.Gs_b_k = None           # column of G^s_b belonging to the prescribed zDot (5,)
        self.eta = None              # general velocity level solution
        self.etaDot = None           # general acceleration level solution
        self.omega_b = None          # angular velocity of the wheel carrier (general)
        self.v_b = None              # velocity of the wheel carrier (general)
        self.omega_b_vert = None     # angular velocity for a pure vertical motion
        self.v_b_vert = None         # velocity for a pure vertical motion
        self.omega_d = None          # angular velocity of the bell crank

        # --- suspension characteristics --------------------------------------
        self.toe = None
        self.camber = None
        self.caster = None
        self.kpi = None
        self.trail = None
        self.scrub = None
        self.caster_trail = None
        self.kingpin_offset_hub = None
        self.rot_axis_low = None     # approximated instantaneous steering axis (plot)
        self.rot_axis_high = None
        self.i_steer = None          # steering ratio omega_a / omega_b

        self.r_lw = None              # longitudinal instant center (LIC)
        self.r_l_rel = None
        self.phi_l = None            # LIC angle
        self.lic_pure_translation = True
        self.r_tw = None              # transverse instant center (TIC)
        self.r_t_rel = None
        self.phi_r = None            # RC angle
        self.h_RC = None             # RC height
        self.rc_pure_translation = True

        self.v_spring_damper = None  # velocity in the spring/damper element
        self.i_MR = None             # motion ratio

        # --- forces and moments ----------------------------------------------
        self.F_i = None              # link forces F_0 ... F_5
        self.i_F_i = None            # link forces related to the wheel load
        self.F_d1_norm = None        # scalar spring/damper force (bell crank)
        self.F_d1 = None
        self.i_F_d1 = None
        self.F_bearing = None        # bell crank bearing force
        self.M_bearing = None        # bell crank bearing moment
        self.i_F_bearing = None      # bell crank bearing forces related to the wheel load
        self.i_M_bearing = None      # bell crank bearing moment related to the wheel load
        self.F_in = None             # resulting force at the inner knuckle
        self.M_in = None             # resulting moment at the inner knuckle
        self.i_F_in = None           # resulting force at the inner knuckle related to the wheel load
        self.M_in_axis = None        # moment component about the rotation axis u_a
        self.i_M_axis = None         # moment component about the rotation axis u_a related to the wheel load
        self.M_in_res = None         # residual (bending) moment carried by the bearings
        self.i_M_res = None          # residual (bending) moment carried by the bearings related to the wheel load

    # =========================================================================
    # BLOCK 1: ROTATION MATRICES AND CLOSURE CONDITIONS (POSITION LEVEL)
    # =========================================================================

    @staticmethod
    def _T_0a(phi, u):
        """
        Rotation matrix {}^{0a}T = {}^0R_a(p_a(u_a, phi_a)) built from the Euler
        parameters (unit quaternion) of the rotation vector (u, phi).
        Used for the rotatable chassis hardpoints (body a) and for the bell crank (body d).
        """
        phi = float(np.ravel(np.asarray(phi, dtype=float))[0])
        if phi == 0.0:
            return np.eye(3)
        # p_tilde is the skew symmetric matrix of the vector part of the quaternion
        p_tilde = np.cross(np.eye(3), np.asarray(u, dtype=float) * np.sin(phi * 0.5))
        return np.eye(3) + 2.0 * np.cos(phi * 0.5) * p_tilde + 2.0 * p_tilde @ p_tilde

    @staticmethod
    def _T_0b(phi, theta, psi):
        """Rotation matrix {}^{0b}T built from the Cardan angles phi, theta, psi."""
        c_ph, c_th, c_ps = np.cos(phi), np.cos(theta), np.cos(psi)
        s_ph, s_th, s_ps = np.sin(phi), np.sin(theta), np.sin(psi)
        return np.array([
            [c_ps*c_th, -s_ps*c_ph + c_ps*s_th*s_ph, s_ps*s_ph + c_ps*s_th*c_ph],
            [s_ps*c_th, c_ps*c_ph + s_ps*s_th*s_ph, -c_ps*s_ph + s_ps*s_th*c_ph],
            [-s_th, c_th*s_ph, c_th*c_ph],
            ])

    def _c_b_func(self, phi, theta, psi):
        """c_b = {}^{0b}T(phi, theta, psi) * {}^b c_b for all wheel carrier hardpoints."""
        return (self._T_0b(phi, theta, psi) @ self.b_c_b.T).T

    def _l_i_func(self, beta, z):
        """
        Distance vectors of the five links,
            l_i = P_b,i - P_a,i = r_b + c_b,i - P_a,i
        evaluated for the primary joint coordinates beta = [phi, theta, psi, x, y].
        """
        phi, theta, psi, x, y = beta
        c_b = self._c_b_func(phi, theta, psi)
        return np.array([x, y, z]) + c_b[:5] - self.P_a[:5]

    def _g_s_i(self, beta, z):
        """
        Implicit closure conditions of the five links on position level,
            g^s = 1/2 * (l^T l - l_const^2) = 0
        Residual vector (5,) handed to least_squares.
        """
        l_i = self._l_i_func(beta, z)
        return 0.5 * (np.sum(l_i ** 2, axis=1) - self.l_i_const ** 2)

    def _c_d_func(self, phi_d, u_d):
        """c_d = {}^{0d}T(u_d, phi_d) * {}^d c_d for both bell crank hardpoints."""
        return (self._T_0a(phi_d, u_d) @ self.d_c_d.T).T

    def _l_d0_func(self, phi_d, u_d):
        """Distance vector of the push-/pullrod, l_d,0 = P_d,0 - P_b,5."""
        c_d = self._c_d_func(phi_d, u_d)
        return self.r_d + c_d[0] - self.P_b[5]

    def _g_s_d(self, phi_d, u_d):
        """Closure condition of the push-/pullrod on position level (scalar)."""
        l_d0 = self._l_d0_func(phi_d, u_d)
        return 0.5 * (l_d0 @ l_d0 - self.l_d_const ** 2)

    # =========================================================================
    # BLOCK 2: GEOMETRY SETUP (called on every steering update)
    # =========================================================================

    def set_geometry(self, P_aP0, P_bP0, P_dP0, r_aP0, r_aDelta, r_bP0, r_dP0, phi_a, u_a):
        """
        Set up all quantities that only depend on the steering input.

        P_aP0    : chassis side hardpoints in design position P0 (6, 3)
        P_bP0    : wheel carrier hardpoints in design position P0 (7, 3)
        P_dP0    : bell crank hardpoints in design position P0 (2, 3)
        r_aP0    : origins of the body fixed frames K_a in design position (5, 3)
        r_aDelta : translational steering offset of the chassis hardpoints (5, 3)
        r_bP0    : origin of the body fixed frame K_b in design position (wheel center)
        r_dP0    : origin of the body fixed frame K_d (bell crank pivot)
        phi_a    : rotation angle of each chassis hardpoint about its axis u_a (5,)
        u_a      : rotation axis of each chassis hardpoint (5, 3)
        """
        P_aP0 = np.asarray(P_aP0, dtype=float)
        P_bP0 = np.asarray(P_bP0, dtype=float)
        P_dP0 = np.asarray(P_dP0, dtype=float)
        r_aP0 = np.asarray(r_aP0, dtype=float)
        r_bP0 = np.asarray(r_bP0, dtype=float)
        r_dP0 = np.asarray(r_dP0, dtype=float)
        u_a = np.asarray(u_a, dtype=float)
        # phi_a may be handed over as (5,) or (5, 1), flatten it for both cases
        phi_a = np.asarray(phi_a, dtype=float).reshape(-1)

        # body fixed lever arms in the design position (z = 0):
        #   {}^a c_a = P_a,P0 - r_a,P0 ,  {}^b c_b = P_b,P0 - r_b,P0 ,  {}^d c_d = P_d,P0 - r_d,P0
        self.a_c_a = P_aP0[:5] - r_aP0
        self.b_c_b = P_bP0 - r_bP0
        self.d_c_d = P_dP0 - r_dP0

        # constant link lengths in the design position (z = 0)
        self.l_i_const = np.linalg.norm(P_bP0[:5] - P_aP0[:5], axis=1)
        self.l_d_const = np.linalg.norm(P_dP0[0] - P_bP0[5])

        # rotate the chassis side lever arms into the inertial frame K_0:
        #   c_a = {}^{0a}T * {}^a c_a
        self.c_a = np.array([self._T_0a(phi_a[i], u_a[i]) @ self.a_c_a[i] for i in range(5)])

        # current chassis hardpoints; the fixed spring/damper mounting point is index 5
        P_a = r_aP0 + np.asarray(r_aDelta, dtype=float) + self.c_a
        self.P_a = np.vstack([P_a, P_aP0[5]])

        self.r_d = r_dP0

    # =========================================================================
    # BLOCK 3: POSITION LEVEL
    # =========================================================================

    def solve_position(self, beta_start, z):
        """
        Solve the nonlinear closure conditions on position level for the primary
        joint coordinates beta at the prescribed ride height z
        (least_squares, Levenberg-Marquardt).
        """
        res = least_squares(self._g_s_i, np.array(beta_start, dtype=float),
                            args=(z,), method='lm')
        if not res.success:
            print(f"Wheel carrier: least_squares warning! Reason: {res.message} Position: {z}")
        self.beta = res.x.copy()
        # r_b = [x, y, z] : position of the wheel carrier frame origin
        self.r_b = np.concatenate((self.beta[3:5], [z]))

    def calc_position(self, z, tire_radius):
        """
        Evaluate all position level quantities for the converged beta:
        distance vectors l_i, lever arms c_b, hardpoints P_b and the vector r_wb
        pointing from the wheel center to the wheel contact patch.

        tire_radius : tire radius
        """
        self.l_i = self._l_i_func(self.beta, z)
        self.c_b = self._c_b_func(*self.beta[:3])
        # wheel carrier hardpoints incl. spring/damper (5) and wheel axis point (6)
        self.P_b = self.r_b + self.c_b

        # c_b[6] is the wheel rotation axis. r_wb is the vector from wheel center
        # to wheel contact patch
        c_b6 = self.c_b[6]
        if abs(c_b6[2]) < self.EPS:
            self.r_wb = tire_radius * np.array([0.0, 0.0, -1.0])
        else:
            z_val = -(c_b6[0] ** 2 + c_b6[1] ** 2) / c_b6[2]
            r_wb_dir = np.concatenate((c_b6[:2], [z_val]))
            self.r_wb = tire_radius / np.linalg.norm(r_wb_dir) * r_wb_dir
            # flip the vector if the wheel axis points in negative z direction
            if c_b6[2] < 0.0:
                self.r_wb = -self.r_wb

    def solve_bell_crank(self, phi_d_start, u_d):
        """
        Solve the closure condition of the push-/pullrod for the primary joint
        coordinate phi_d (rotation angle of the bell crank about its axis u_d).
        Requires an up to date calc_position(), because P_b,5 is used.
        """
        res = least_squares(self._g_s_d, np.atleast_1d(np.asarray(phi_d_start, dtype=float)),
                            args=(u_d,), method='lm')
        if not res.success:
            print(f"Bell crank: least_squares warning! Reason: {res.message} "
                  f"Position: {self.r_b[2]}")
        self.phi_d = float(res.x[0])

    def calc_bell_crank(self, u_d):
        """Evaluate lever arms, hardpoints and push-/pullrod vector of the bell crank."""
        self.c_d = self._c_d_func(self.phi_d, u_d)
        self.P_d = self.r_d + self.c_d

        # push-/pullrod; l_5 is the same rod written in the P_b - P_a convention
        # that is used for the link forces
        self.l_d0 = self.P_d[0] - self.P_b[5]
        self.l_5 = -self.l_d0

        # spring/damper element
        self.l_d1 = self.P_d[1] - self.P_a[5]
        self.spring_damper_length = np.linalg.norm(self.l_d1)

    def calc_inner_axis_display(self, u_a, r_aP0, P_aP0):
        """
        Upper and lower point of the inner knuckle rotation axis, only needed for
        the 3d plot. All rotatable chassis hardpoints share the same axis.
        """
        P_aP0 = np.asarray(P_aP0, dtype=float)
        z_top = P_aP0[:5, 2].max() + 50.0
        z_bot = P_aP0[:5, 2].min() - 50.0
        r_a = np.asarray(r_aP0, dtype=float)[0]
        u = np.asarray(u_a, dtype=float)[0]
        self.P_axis_a_upper = r_a + (z_top - r_a[2]) / u[2] * u
        self.P_axis_a_lower = r_a + (z_bot - r_a[2]) / u[2] * u

    # =========================================================================
    # BLOCK 4: VELOCITY AND ACCELERATION LEVEL
    # =========================================================================

    def calc_Gs_a(self):
        """
        Constraint matrix of body a (chassis side):
            G^s_a = [ l^T c~_a   -l^T ]   ->   [ l x c_a , -l ]
        """
        self.Gs_a = np.hstack([np.cross(self.l_i, self.c_a), -self.l_i])

    def calc_Gs_b(self):
        """
        Constraint matrix of body b (wheel carrier):
            G^s_b = [ -l^T c~_b   l^T ]   ->   [ -(l x c_b) , l ]
        Split into the part A belonging to the unknowns [omega_b, v_bx, v_by]
        and the column k belonging to the prescribed coordinate zDot.
        """
        Gs_b = np.hstack([-np.cross(self.l_i, self.c_b[:5]), self.l_i])
        self.Gs_b_A = Gs_b[:, :5]
        self.Gs_b_k = Gs_b[:, 5]

    def _solve_eta(self, zDot, mode, omega_a, v_a):
        """
        Solve the linear closure conditions on velocity level for
        eta = [omega_bx, omega_by, omega_bz, v_bx, v_by] and return it.
        The mode selects which excitations contribute to the right hand side.
        """
        eta_a = np.hstack([np.asarray(omega_a, dtype=float), np.asarray(v_a, dtype=float)])
        Gs_a_eta_a = np.sum(self.Gs_a * eta_a, axis=1)  # row wise scalar product

        if mode == self.MODE_FULL:
            b = -self.Gs_b_k * zDot - Gs_a_eta_a
        elif mode == self.MODE_HEAVE:
            b = -self.Gs_b_k * zDot
        elif mode == self.MODE_STEER:
            b = -Gs_a_eta_a
        else:
            raise ValueError(f"unknown velocity level mode: {mode}")

        return np.linalg.solve(self.Gs_b_A, b)

    def calc_eta(self, zDot, omega_a, v_a, mode=MODE_FULL):
        """
        Solve and store the velocity level solution together with the resulting
        wheel carrier velocities omega_b and v_b.
        """
        self.eta = self._solve_eta(zDot, mode, omega_a, v_a)
        self.omega_b = self.eta[:3]
        self.v_b = np.concatenate((self.eta[3:5], [zDot]))
        return self.eta

    def calc_etaDot(self, zDot, z2Dot, omega_a, v_a, omegaDot_a, vDot_a):
        """
        Solve the closure conditions on acceleration level for
        etaDot = [omegaDot_b, vDot_bx, vDot_by].
        Requires calc_eta(..., MODE_FULL) to be up to date.
        """
        omega_a = np.asarray(omega_a, dtype=float)
        v_a = np.asarray(v_a, dtype=float)
        etaDot_a = np.hstack([np.asarray(omegaDot_a, dtype=float),
                              np.asarray(vDot_a, dtype=float)])
        Gs_a_etaDot_a = np.sum(self.Gs_a * etaDot_a, axis=1)

        # gamma^s_ab of the acceleration level
        cDot_b = np.cross(self.omega_b, self.c_b[:5])
        cDot_a = np.cross(omega_a, self.c_a)
        lDot_i = self.v_b + cDot_b - v_a - cDot_a
        gamma_s_ab = (np.sum(self.l_i * (np.cross(self.omega_b, cDot_b)
                                         - np.cross(omega_a, cDot_a)), axis=1)
                      + np.sum(lDot_i * lDot_i, axis=1))

        b = -self.Gs_b_k * z2Dot - gamma_s_ab - Gs_a_etaDot_a
        self.etaDot = np.linalg.solve(self.Gs_b_A, b)
        return self.etaDot

    def calc_bell_crank_omega(self, u_d):
        """
        Closure condition of the push-/pullrod on velocity level, solved for the
        angular velocity scalar omega_d of the bell crank:
            omega_d = ([l^T c~_b  -l^T] [omega_b; v_b]) / (l^T c~_d u_d)
        Evaluated for a pure vertical motion, i.e. with omega_b_vert / v_b_vert.
        """
        Gs_b_rod = np.hstack([-np.cross(self.l_d0, self.c_b[5]), self.l_d0])
        Gs_d_rod = np.cross(self.l_d0, self.c_d[0])
        eta_b = np.concatenate((self.omega_b_vert, self.v_b_vert))
        omega_d_scalar = -np.dot(Gs_b_rod, eta_b) / np.dot(Gs_d_rod, u_d)
        self.omega_d = omega_d_scalar * u_d

    # =========================================================================
    # BLOCK 5: SUSPENSION CHARACTERISTICS
    # =========================================================================

    @staticmethod
    def _toe_from_wheel_axis(c_b6):
        """Toe angle from the wheel rotation axis, wrapped to the range ]-pi, pi]."""
        raw_toe = -(np.arctan2(c_b6[1], c_b6[0]) + 0.5 * np.pi)
        return (raw_toe + np.pi) % (2.0 * np.pi) - np.pi

    def calc_kinematic_properties(self, omega_a, v_a, has_bell_crank):
        """
        Camber, toe and all steering axis related characteristics. The steering
        axis is approximated by the instantaneous screw axis of a pure steering
        motion. Trail, scrub, caster trail and kingpin offset are rotated with
        the wheel (toe), in contrast to MSC Adams/Car.
        """
        # --- camber and toe from the wheel rotation axis c_b,6 ---------------
        c_b6 = self.c_b[6]
        self.camber = 0.5 * np.pi - np.arccos(c_b6 @ np.array([0.0, 0.0, 1.0])
                                              / np.linalg.norm(c_b6))
        self.toe = self._toe_from_wheel_axis(c_b6)
        T_toe = self._T_0b(0.0, 0.0, self.toe)

        # direct spring/damper: the length is simply the distance of its mounting points
        if not has_bell_crank:
            self.spring_damper_length = np.linalg.norm(self.P_a[5] - self.P_b[5])

        # --- instantaneous screw axis of a pure steering motion --------------
        eta_steer = self._solve_eta(0.0, self.MODE_STEER, omega_a, v_a)
        omega_b_steer = eta_steer[:3]
        v_b_steer = np.concatenate((eta_steer[3:5], [0.0]))

        if np.linalg.norm(omega_b_steer) < self.EPS:
            # pure translation: no steering axis exists, keep the previous values
            print("Warning: pure translation, steering axis is undefined")
            return

        # a point on the instantaneous screw axis
        r_P = self.r_b + np.cross(omega_b_steer, v_b_steer) / np.linalg.norm(omega_b_steer) ** 2

        # pierce points of the screw axis with the horizontal planes through the
        # wheel center (b) and through the wheel contact patch (w)
        t = (self.r_b[2] - r_P[2]) / omega_b_steer[2]
        r_b_pierce = r_P + t * omega_b_steer
        t = (self.r_b[2] + self.r_wb[2] - r_P[2]) / omega_b_steer[2]
        r_w_pierce = r_P + t * omega_b_steer

        # rotate the lever arms with the wheel before evaluating the characteristics
        b_pierce_rot = T_toe @ (self.r_b - r_b_pierce)
        w_pierce_rot = T_toe @ (self.r_b + self.r_wb - r_w_pierce)

        # approximated instantaneous steering axis (only used for the 3d plot)
        self.rot_axis_low = r_w_pierce
        self.rot_axis_high = 2.0 * (r_b_pierce - r_w_pierce) + r_w_pierce

        self.trail = -w_pierce_rot[0]
        self.scrub = w_pierce_rot[1]
        self.caster_trail = -b_pierce_rot[0]
        self.kingpin_offset_hub = b_pierce_rot[1]

        # caster and kingpin inclination from the direction of the steering axis.
        # Not using the angular velocity directly because otherwise both angles
        # would change sign together with the direction of the steering velocity.
        # The axis is rotated with the wheel as well.
        u_steer = T_toe @ (r_b_pierce - r_w_pierce)
        u_steer_xz = np.array([u_steer[0], 0.0, u_steer[2]])
        self.caster = -(0.5 * np.pi - np.arccos(u_steer_xz @ np.array([1.0, 0.0, 0.0])
                                                / np.linalg.norm(u_steer_xz)))
        u_steer_yz = np.array([0.0, u_steer[1], u_steer[2]])
        self.kpi = -(0.5 * np.pi - np.arccos(u_steer_yz @ np.array([0.0, 1.0, 0.0])
                                             / np.linalg.norm(u_steer_yz)))

    def calc_steering_ratio(self, omega_a, v_a, steerDot):
        """
        Steering ratio i_steer = omega_a / omega_b, i.e. the ratio of the angular
        velocity magnitudes of inner knuckle and wheel carrier during a pure steering
        motion. The toe angle is updated as well (needed for the toe over steer plot).
        """
        self.toe = self._toe_from_wheel_axis(self.c_b[6])
        omega_b_steer = self._solve_eta(0.0, self.MODE_STEER, omega_a, v_a)[:3]
        self.i_steer = steerDot / np.linalg.norm(omega_b_steer)

    def calc_instant_centers(self, zDot):
        """
        Longitudinal instant center (LIC) and roll center (RC), both evaluated for
        a pure vertical motion of the wheel carrier. Also stores omega_b_vert and
        v_b_vert, which are reused for the spring/damper velocity and the bell crank.
        """
        eta_vert = self._solve_eta(zDot, self.MODE_HEAVE, np.zeros((5, 3)), np.zeros((5, 3)))
        self.omega_b_vert = eta_vert[:3]
        self.v_b_vert = np.concatenate((eta_vert[3:5], [zDot]))
        omega_b, v_b = self.omega_b_vert, self.v_b_vert
        r_w = self.r_b + self.r_wb  # wheel contact patch

        # ---------------------------------------------------------------------
        # Longitudinal instant center (LIC), projection into the x-z-plane
        # ---------------------------------------------------------------------
        self.lic_pure_translation = abs(omega_b[1]) < self.EPS
        if not self.lic_pure_translation:
            # analytic simplification of r_l = r_b + (omega x v) / |omega|^2
            r_l = self.r_b + np.array([v_b[2] / omega_b[1], 0.0, -v_b[0] / omega_b[1]])

            # direction vector from the wheel contact patch to the LIC
            self.r_lw = r_l - r_w

            # the LIC may lie in front of or behind the wheel, therefore the line
            # is evaluated at the fixed position x = r_w,x - 1000
            t = -1000.0 / self.r_lw[0]
            self.r_l_rel = t * self.r_lw

            # projection into the x-z-plane and LIC angle phi_l
            r_l_rel_xz = np.array([self.r_l_rel[0], 0.0, self.r_l_rel[2]])
            self.phi_l = np.arcsin(self.r_l_rel[2] / np.linalg.norm(r_l_rel_xz))

        # ---------------------------------------------------------------------
        # Roll center (RC), projection into the y-z-plane
        # ---------------------------------------------------------------------
        self.rc_pure_translation = abs(omega_b[0]) < self.EPS
        if not self.rc_pure_translation:
            # analytic simplification of r_t = r_b + (omega x v) / |omega|^2
            r_t = self.r_b + np.array([0.0, -v_b[2] / omega_b[0], v_b[1] / omega_b[0]])

            # direction vector from the wheel contact patch to the TIC
            self.r_tw = r_t - r_w

            # for a symmetric axle and equal sided bump the RC lies at y = 0
            t = -r_w[1] / self.r_tw[1]
            self.r_t_rel = t * self.r_tw
            self.h_RC = t * self.r_tw[2] + r_w[2]

            # projection into the y-z-plane and RC angle phi_r
            r_t_rel_yz = np.array([0.0, self.r_t_rel[1], self.r_t_rel[2]])
            self.phi_r = np.arcsin(self.r_t_rel[2] / np.linalg.norm(r_t_rel_yz))

    def calc_bell_crank_properties(self, u_d, zDot, F_w):
        """
        Spring/damper velocity and motion ratio, spring/damper force and the
        resulting bearing loads of the bell crank.
        Requires calc_link_forces() (uses the push-/pullrod force F_5).
        """
        l_d1_hat = self.l_d1 / np.linalg.norm(self.l_d1)
        l_5_hat = self.l_5 / np.linalg.norm(self.l_5)

        # v_d,1 = omega_d x c_d,1, projected onto the spring/damper direction
        v_d1 = np.cross(self.omega_d, self.c_d[1])
        self.v_spring_damper = np.dot(v_d1, l_d1_hat)
        self.i_MR = np.abs(self.v_spring_damper / zDot)

        # moment equilibrium about the bell crank axis u_d -> spring/damper force
        F_w_norm = np.linalg.norm(F_w)
        F_5_vec = self.F_i[5] * l_5_hat
        numerator = np.cross(self.c_d[0], F_5_vec) @ u_d
        denominator = np.cross(self.c_d[1], l_d1_hat) @ u_d
        self.F_d1_norm = -numerator / denominator
        self.i_F_d1 = np.abs(self.F_d1_norm) / F_w_norm
        self.F_d1 = self.F_d1_norm * l_d1_hat

        # force and moment equilibrium at the bell crank pivot
        self.F_bearing = -(F_5_vec + self.F_d1)
        M_5 = np.cross(self.c_d[0], F_5_vec)
        M_d1 = np.cross(self.c_d[1], self.F_d1)
        self.M_bearing = -(M_5 + M_d1)

        self.i_F_bearing = np.linalg.norm(self.F_bearing) / F_w_norm
        self.i_M_bearing = np.linalg.norm(self.M_bearing) / F_w_norm

    # =========================================================================
    # BLOCK 6: FORCE AND MOMENT CALCULATION
    # =========================================================================

    def calc_link_forces(self, F_w, F_t, has_bell_crank, zDot):
        """
        Link forces F_0 ... F_5 from the force and moment equilibrium of the wheel
        carrier about the wheel center r_b. All links are modelled as two force
        members, index 5 is the spring/damper element resp. the push-/pullrod.

        F_w : force at the wheel contact patch
        F_t : additional force at the wheel center
        """
        if not has_bell_crank:
            # direct spring/damper element between wheel carrier and chassis
            self.l_5 = self.P_b[5] - self.P_a[5]
        l_all = np.vstack((self.l_i, self.l_5))

        # e = -l / ||l|| : unit vectors in link direction (6, 3)
        E = -l_all / np.linalg.norm(l_all, axis=1, keepdims=True)
        # xi = c_b x e : unit moment vectors (6, 3)
        Xi = np.cross(self.c_b[:6], E)
        # coefficient matrix of the linear system (6, 6)
        E_Xi = np.vstack((E.T, Xi.T))

        F_w = np.asarray(F_w, dtype=float)
        F_t = np.asarray(F_t, dtype=float)
        # moment of the contact patch force about the wheel center
        M_w = np.cross(self.r_wb, F_w)
        rhs = np.concatenate((-F_w - F_t, -M_w))

        # solve the linear system for the link forces
        self.F_i = np.linalg.solve(E_Xi, rhs)
        F_w_norm = np.linalg.norm(F_w)
        self.i_F_i = self.F_i / F_w_norm

        if not has_bell_crank:
            # spring/damper velocity and motion ratio of the direct configuration.
            # The minus sign compensates the l_5 = P_b,5 - P_a,5 convention: the
            # ratio is defined positive if P_a,5 lies above P_b,5 (see documentation).
            l_5_hat = -self.l_5 / np.linalg.norm(self.l_5)
            v_b5 = np.cross(self.omega_b_vert, self.c_b[5]) + self.v_b_vert
            self.v_spring_damper = np.dot(v_b5, l_5_hat)
            self.i_MR = np.abs(self.v_spring_damper / zDot)

    def calc_inner_knuckle_loads(self, u_a, a_i, F_w):
        """
        Resulting force F_in and moment M_in at the inner knuckle, split into the
        component about the rotation axis u_a and the residual bending moment that
        has to be carried by the bearings.

        a_i : selection vector, marks the links connected to the inner knuckle
        """
        u = np.asarray(u_a, dtype=float)[0]  # identical for all inner knuckle hardpoints
        a_mask = np.asarray(a_i, dtype=float)[:5, np.newaxis]
        F_w_norm = np.linalg.norm(F_w)

        # unit vectors of the links, here without the minus sign because the
        # reaction forces at the chassis side act in the opposite direction
        L_hat = self.l_i / np.linalg.norm(self.l_i, axis=1, keepdims=True)
        F_vec = self.F_i[:5, np.newaxis] * L_hat * a_mask
        M_vec = np.cross(self.c_a[:5], F_vec)

        self.F_in = np.sum(F_vec, axis=0)
        self.M_in = np.sum(M_vec, axis=0)
        self.i_F_in = np.linalg.norm(self.F_in) / F_w_norm

        # moment component about the rotation axis and residual bearing moment
        self.M_in_axis = np.dot(self.M_in, u) * u
        self.i_M_axis = np.linalg.norm(self.M_in_axis) / F_w_norm
        self.M_in_res = self.M_in - self.M_in_axis
        self.i_M_res = np.linalg.norm(self.M_in_res) / F_w_norm

    # =========================================================================
    # BLOCK 7: DRIVER METHODS FOR THE PLOTS
    # =========================================================================

    def calc_geometry(self, z, zDot, beta_start, phi_d_start, omega_a, v_a, tire_radius, u_d,
                      has_bell_crank, F_w, F_t):
        """Complete evaluation of one suspension position (3d plot and characteristics)."""
        self.solve_position(beta_start, z)
        self.calc_position(z, tire_radius)
        if has_bell_crank:
            self.solve_bell_crank(phi_d_start, u_d)
            self.calc_bell_crank(u_d)
        self.calc_Gs_a()
        self.calc_Gs_b()
        self.calc_kinematic_properties(omega_a, v_a, has_bell_crank)
        self.calc_instant_centers(zDot)
        if has_bell_crank:
            self.calc_bell_crank_omega(u_d)
        self.calc_link_forces(F_w, F_t, has_bell_crank, zDot)
        if has_bell_crank:
            self.calc_bell_crank_properties(u_d, zDot, F_w)

    def calc_steering_plots(self, z, steerDot, beta_start, phi_d_start, omega_a, v_a,
                            tire_radius, u_d, has_bell_crank):
        """
        Reduced evaluation of one steering position for the characteristics over
        the steering input (no instant centers, no forces). The bell crank is
        solved as well, so that the spring/damper length is also correct for the
        bell crank configuration.
        """
        self.solve_position(beta_start, z)
        self.calc_position(z, tire_radius)
        if has_bell_crank:
            self.solve_bell_crank(phi_d_start, u_d)
            self.calc_bell_crank(u_d)
        self.calc_Gs_a()
        self.calc_Gs_b()
        self.calc_steering_ratio(omega_a, v_a, steerDot)
        self.calc_kinematic_properties(omega_a, v_a, has_bell_crank)


class LiveMovementBase:
    """
    Common part of the interactive suspension simulations.

    The class owns the kinematics object, the axle data, the calculated
    characteristics and both windows: the 3d view of the suspension geometry and
    the 2d characteristic curves. It also contains the complete sweep logic over
    ride height and steering input.

    The subclasses only describe HOW the steering input acts on the chassis
    hardpoints, by implementing the hooks of BLOCK 8:

        LiveMovementStd5Link          : translational steering input
        LiveMovementRotationAxis5Link : rotational steering input (inner knuckle)

    Typical call order:
        set_axle_data()        -> geometry and hardware of the axle
        set_simulation_data()  -> ranges, prescribed velocities and loads
        calc_start_values()    -> coarse start value matrix over z and steering
        setup_figures()        -> create both windows
        add_controls()         -> sliders, buttons and the first update
    """

    # True for subclasses whose chassis hardpoints rotate about an inner knuckle
    # axis. Only for those the inner knuckle loads and the steering ratio exist.
    has_rotating_knuckle = False

    # True if the steering input is an angle. Only then the characteristics over
    # the steering input are calculated, their x axis is plotted in degrees.
    sweeps_steering = False

    # Definition of the 2d characteristic plots: (result key, unit, x axis).
    # The key is used as plot label and as key of the results dictionaries,
    # the x axis is either the ride height "z" or the steering input "steer".
    # Keys with x axis "z" are read from results, keys with x axis "steer"
    # from results_over_steer.
    PLOT_CONFIGS = [
        ("toe", "°", "z"),
        ("toe_over_steer", "°", "steer"),
        ("steering_ratio", "-", "z"),
        ("steering_ratio_over_steer", "-", "steer"),
        ("camber", "°", "z"),
        ("caster", "°", "z"),
        ("kpi", "°", "z"),
        ("trail", "mm", "z"),
        ("scrub", "mm", "z"),
        ("caster_trail", "mm", "z"),
        ("kingpin_offset_hub", "mm", "z"),
        ("hub_x", "mm", "z"),
        ("hub_y", "mm", "z"),
        ("pitch_center_angle", "°", "z"),
        ("roll_center_angle", "°", "z"),
        ("roll_center_height", "mm", "z"),
        ("spring_damper_length", "mm", "z"),
        ("spring_damper_velocity_ratio", "-", "z"),
        ("spring_damper_force_ratio", "-", "z"),
        ("F_bearing_ratio", "-", "z"),
        ("M_bearing_ratio", "mm", "z"),
        ("F_push_pull_rod_ratio", "-", "z"),
        ("F_0_ratio", "-", "z"),
        ("F_1_ratio", "-", "z"),
        ("F_2_ratio", "-", "z"),
        ("F_3_ratio", "-", "z"),
        ("F_4_ratio", "-", "z"),
        ("F_sum_inner_ratio", "-", "z"),
        ("M_inner_axis_ratio", "mm", "z"),
        ("M_inner_residual_ratio", "mm", "z"),
        ]

    # all keys stored per suspension position over the ride height z,
    # beta is collected but not plotted
    RESULT_KEYS = ["beta"] + [key for key, _unit, axis in PLOT_CONFIGS if axis == "z"]

    # all keys stored per steering position at the current ride height. Only the
    # keys that also appear in PLOT_CONFIGS are plotted, all others are only
    # calculated, so that they are available in the variable explorer.
    # steer is the steering input in degrees, it is the x axis of these curves.
    STEER_RESULT_KEYS = [
        "steer",
        "beta",
        "steering_ratio_over_steer",
        "toe_over_steer",
        "camber_over_steer",
        "caster_over_steer",
        "kpi_over_steer",
        "trail_over_steer",
        "scrub_over_steer",
        "spring_damper_length_over_steer",
        "hub_x",
        "hub_y",
        "x_contact_patch",
        "y_contact_patch",
        ]

    # number of rows and columns of the 2d subplot grid
    PLOT_ROWS = 5
    PLOT_COLS = 6

    # refinement factor of the fine sweep grid compared to the coarse start value grid
    FINE_FACTOR = 5

    def __init__(self, kine_object):
        self.kine = kine_object

        # --- axle data, filled by set_axle_data() ----------------------------
        self.P_aP0 = None              # chassis hardpoints in design position (6, 3)
        self.P_bP0 = None              # wheel carrier hardpoints in design position (7, 3)
        self.P_dP0 = None              # bell crank hardpoints in design position (2, 3)
        self.r_aP0 = None              # origins of the body fixed frames K_a (5, 3)
        self.r_bP0 = None              # origin of the body fixed frame K_b (wheel center)
        self.r_dP0 = None              # bell crank pivot
        self.a_i = None                # selection vector, marks the steered links
        self.u_a = None                # rotation axes of the chassis hardpoints (5, 3)
        self.u_d = None                # rotation axis of the bell crank
        self.has_bell_crank = False    # bell crank instead of a direct spring/damper
        self.link_diameters = None     # diameters of the five links and of rod 5
        self.bearing_sizes = None      # [[z_start, z_end, diameter], ...] of the bearings
        self.tire_radius = None        # tire radius
        self.tire_width = None

        # --- simulation data, filled by set_simulation_data() ----------------
        self.z_min = None              # ride height range and step width
        self.z_max = None
        self.z_step = None
        self.steer_min = None          # steering range and step width in rad
        self.steer_max = None
        self.steer_step = None
        self.zDot = None               # prescribed vertical velocity and acceleration
        self.z2Dot = None
        self.steerDot = None           # prescribed steering velocity and acceleration
        self.steer2Dot = None
        self.F_w = None                # force at the wheel contact patch
        self.F_t = None                # additional force at the wheel center

        # --- calculated characteristics --------------------------------------
        # results            : characteristics over the ride height z
        # results_over_steer : characteristics over the steering input at the
        #                      current ride height, only filled if the subclass
        #                      sweeps the steering angle
        self.results = {key: None for key in self.RESULT_KEYS}
        self.results_over_steer = {key: None for key in self.STEER_RESULT_KEYS}

        # --- sweep grids -----------------------------------------------------
        self.z_array = None            # coarse ride height grid (start value matrix)
        self.steer_array = None        # coarse steering grid (start value matrix)
        self.z_array_fine = None       # fine ride height grid (characteristic curves)
        self.steer_array_fine = None   # fine steering grid in degrees (characteristic curves)

        # --- start values of the nonlinear solvers ---------------------------
        self.beta_start = None         # start vector of the primary joint coordinates
        self.beta_start_z0 = None      # start vector at z = 0, reused per steering step
        self.beta_list = None          # beta over z for one steering position
        self.beta_matrix = None        # beta over z and steering (3d matrix)
        self.phi_d_start = 0.0         # start value of the bell crank angle phi_d
        self.phi_d_start_z0 = 0.0
        self.phi_d_list = None         # phi_d over z for one steering position
        self.phi_d_matrix = None       # phi_d over z and steering (2d matrix)

        # --- current slider position -----------------------------------------
        self.z_curr = 0.0
        self.steer_curr = 0.0
        self.z_index = 0               # index of the closest coarse z value
        self.z0_index = 0              # index of the coarse z value closest to z = 0
        self.steer_index = 0           # index of the closest coarse steering value
        self.steer0_index = 0          # index of the steering value closest to zero

        # --- figures and axes ------------------------------------------------
        self.fig_3d = None
        self.ax_3d = None
        self.fig_2d = None
        self.axes_2d = []
        self.xlim = None               # fixed limits of the 3d view
        self.ylim = None
        self.zlim = None
        self.z_slider_3d = None
        self.z_slider_2d = None
        self.steer_slider_3d = None
        self.steer_slider_2d = None
        self.view_button_axes = []     # references, otherwise the buttons stop working
        self.view_buttons = []

        # unit sphere mesh, reused for every joint drawn in the 3d view
        u_sphere = np.linspace(0, 2 * np.pi, 10)
        v_sphere = np.linspace(0, np.pi, 10)
        U_SPHERE, V_SPHERE = np.meshgrid(u_sphere, v_sphere)
        self._sphere_x = np.sin(V_SPHERE) * np.cos(U_SPHERE)
        self._sphere_y = np.sin(V_SPHERE) * np.sin(U_SPHERE)
        self._sphere_z = np.cos(V_SPHERE)



    # =========================================================================
    # BLOCK 8: INPUT DATA AND INTERFACE OF THE SUBCLASSES
    # =========================================================================

    def _store_axle_data(self, P_aP0, P_bP0, P_dP0, r_bP0, r_dP0, a_i, u_d, has_bell_crank,
                         link_diameters, tire_radius, tire_width, bearing_sizes=None):
        """
        Store the axle data that both axle types have in common. Called by
        set_axle_data() of the subclass, which adds its own data on top.

        P_aP0 / P_bP0 / P_dP0 : hardpoints in design position (chassis / wheel
                                carrier / bell crank)
        r_bP0 / r_dP0         : wheel center and bell crank pivot in design position
        a_i                   : selection vector, marks the steered links
        u_d                   : normalized rotation axis of the bell crank
        has_bell_crank        : bell crank instead of a direct spring/damper element
        link_diameters        : diameters of the five links and of spring/damper element or push-/pullrod (3d view)
        tire_radius / tire_width        : tire radius and tire width
        bearing_sizes         : bearings on the inner knuckle axis (3d view)
        """
        self.P_aP0 = np.asarray(P_aP0, dtype=float)
        self.P_bP0 = np.asarray(P_bP0, dtype=float)
        self.P_dP0 = np.asarray(P_dP0, dtype=float)
        self.r_bP0 = np.asarray(r_bP0, dtype=float)
        self.r_dP0 = np.asarray(r_dP0, dtype=float)
        self.a_i = np.asarray(a_i, dtype=float)
        self.u_d = np.asarray(u_d, dtype=float)
        self.has_bell_crank = bool(has_bell_crank)
        self.link_diameters = link_diameters
        self.tire_radius = tire_radius
        self.tire_width = tire_width
        self.bearing_sizes = bearing_sizes

    def set_simulation_data(self, z_min, z_max, z_step, steer_min, steer_max, steer_step,
                            zDot, z2Dot, steerDot, steer2Dot, F_w, F_t):
        """
        Ranges, prescribed velocities and loads of the simulation.

        z_min / z_max / z_step             : ride height range and step width
        steer_min / steer_max / steer_step : steering range and step width
        zDot / z2Dot                       : vertical velocity and acceleration
        steerDot / steer2Dot               : steering velocity and acceleration
        F_w / F_t                          : force at the wheel contact patch and
                                             at the wheel center
        """
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.steer_min = steer_min
        self.steer_max = steer_max
        self.steer_step = steer_step
        self.zDot = zDot
        self.z2Dot = z2Dot
        self.steerDot = steerDot
        self.steer2Dot = steer2Dot
        self.F_w = np.asarray(F_w, dtype=float)
        self.F_t = np.asarray(F_t, dtype=float)

    def set_axle_data(self, *args, **kwargs):
        """Store the axle data, implemented in the subclass."""
        raise NotImplementedError

    def apply_steering(self, steer):
        """
        Hand the geometry of one steering position over to the kinematics object,
        implemented in the subclass.
        """
        raise NotImplementedError

    def steering_velocities(self):
        """
        Velocities and accelerations of the chassis hardpoints that belong to the
        prescribed steering velocity, implemented in the subclass.
        Returns (omega_a, v_a, omegaDot_a, vDot_a).
        """
        raise NotImplementedError

    def calc_extra_characteristics(self, omega_a, v_a):
        """
        Characteristics that only exist for a specific axle type, evaluated for
        every suspension position. Does nothing by default.
        """
        pass

    def calc_display_geometry(self):
        """
        Geometry that is only needed for the 3d view of a specific axle type,
        evaluated once per slider update. Does nothing by default.
        """
        pass

    def draw_custom_geometry(self):
        """
        Hook for additional geometry of a specific axle type, called at the end of
        the 3d view. Does nothing by default.
        """
        pass



    # =========================================================================
    # BLOCK 9: HELPERS FOR THE SWEEPS
    # =========================================================================

    @staticmethod
    def nearest_index(array, value):
        """Index of the entry of array that is closest to value."""
        return int(np.abs(np.asarray(array) - value).argmin())

    @staticmethod
    def new_result_buffers(results):
        """
        Two empty collections with the keys of results, one for the positive and
        one for the negative sweep direction.
        """
        return {k: [] for k in results}, {k: [] for k in results}

    @staticmethod
    def store_sweep_results(results, data_plus, data_minus):
        """
        Merge the two sweep directions into one array per characteristic and
        store them in results (self.results or self.results_over_steer). The
        positive branch is reversed, so that all curves run from the maximum to
        the minimum of the sweep variable.
        """
        for key in results:
            results[key] = np.array(data_plus[key][::-1] + data_minus[key])

    def extract_results(self, z):
        """
        Read all characteristics of the current suspension position out of the
        kinematics object. Quantities that do not exist for the current
        configuration are set to zero, so that all curves have the same length.
        """
        kine = self.kine
        return {
            'beta': np.concatenate((kine.beta, [z])),
            'caster': np.degrees(kine.caster),
            'kpi': np.degrees(kine.kpi),
            'toe': np.degrees(kine.toe),
            'camber': np.degrees(kine.camber),
            'trail': kine.trail,
            'scrub': kine.scrub,
            'caster_trail': kine.caster_trail,
            'kingpin_offset_hub': kine.kingpin_offset_hub,
            'hub_x': kine.r_b[0],
            'hub_y': kine.r_b[1],
            'pitch_center_angle': np.nan if kine.lic_pure_translation else np.degrees(kine.phi_l),
            'roll_center_angle': np.nan if kine.rc_pure_translation else np.degrees(kine.phi_r),
            # shifted by the tire radius, so that the contact patch is the zero level
            'roll_center_height': np.nan if kine.rc_pure_translation else kine.h_RC + self.tire_radius,
            'spring_damper_length': kine.spring_damper_length,
            'spring_damper_velocity_ratio': kine.i_MR,
            'spring_damper_force_ratio': kine.i_F_d1 if self.has_bell_crank else kine.i_F_i[5],
            'F_bearing_ratio': kine.i_F_bearing if self.has_bell_crank else 0.0,
            'M_bearing_ratio': kine.i_M_bearing if self.has_bell_crank else 0.0,
            'F_push_pull_rod_ratio': kine.i_F_i[5] if self.has_bell_crank else 0.0,
            'F_0_ratio': kine.i_F_i[0],
            'F_1_ratio': kine.i_F_i[1],
            'F_2_ratio': kine.i_F_i[2],
            'F_3_ratio': kine.i_F_i[3],
            'F_4_ratio': kine.i_F_i[4],
            'F_sum_inner_ratio': kine.i_F_in if self.has_rotating_knuckle else 0.0,
            'M_inner_axis_ratio': kine.i_M_axis if self.has_rotating_knuckle else 0.0,
            'M_inner_residual_ratio': kine.i_M_res if self.has_rotating_knuckle else 0.0,
            'steering_ratio': kine.i_steer if self.has_rotating_knuckle else 0.0,
            }

    def extract_steer_results(self, steer, z):
        """
        Read all characteristics of the current steering position out of the
        kinematics object, counterpart of extract_results() for the sweep over
        the steering input. The keys have to match STEER_RESULT_KEYS.

        steer : steering input in rad, stored in degrees
        z     : current ride height
        """
        kine = self.kine
        return {
            'steer': np.degrees(steer),
            'beta': np.concatenate((kine.beta, [z])),
            'steering_ratio_over_steer': kine.i_steer,
            'toe_over_steer': np.degrees(kine.toe),
            'camber_over_steer': np.degrees(kine.camber),
            'caster_over_steer': np.degrees(kine.caster),
            'kpi_over_steer': np.degrees(kine.kpi),
            'trail_over_steer': kine.trail,
            'scrub_over_steer': kine.scrub,
            'spring_damper_length_over_steer': kine.spring_damper_length,
            'hub_x': kine.r_b[0],
            'hub_y': kine.r_b[1],
            'x_contact_patch': kine.r_b[0] + kine.r_wb[0],
            'y_contact_patch': kine.r_b[1] + kine.r_wb[1],
            }



    # =========================================================================
    # BLOCK 10: START VALUE MATRIX
    # =========================================================================

    def calc_start_values(self):
        """
        Solve the position level on the coarse grid over ride height and steering
        input. The resulting matrices are used as start vectors for the solvers,
        so that every slider position starts from a converged neighbour solution.
        """
        start_time = time.time()

        # coarse grids, flipped so that all curves run from max to min
        self.z_array = np.flip(np.arange(self.z_min, self.z_max + self.z_step, self.z_step))
        self.steer_array = np.flip(np.arange(self.steer_min, self.steer_max + self.steer_step,
                                             self.steer_step))

        # start vector of the very first sweep, taken from the design position
        self.beta_start = np.concatenate((np.zeros(3), self.r_bP0[:2]))
        self.phi_d_start = 0.0

        beta_plus, beta_minus = [], []
        phi_d_plus, phi_d_minus = [], []

        # sweep from the straight ahead position to both steering ends, every
        # steering position starts from the solution of the previous one at z = 0
        for steer in np.arange(0, self.steer_max + self.steer_step, self.steer_step):
            self._sweep_ride_height(steer)
            beta_plus.append(self.beta_list)
            phi_d_plus.append(self.phi_d_list)

        for steer in np.arange(-self.steer_step, self.steer_min - self.steer_step, -self.steer_step):
            self._sweep_ride_height(steer)
            beta_minus.append(self.beta_list)
            phi_d_minus.append(self.phi_d_list)

        # complete matrices over steering (rows) and ride height (columns)
        beta_plus.reverse()
        phi_d_plus.reverse()
        self.beta_matrix = np.array(beta_plus + beta_minus)
        self.phi_d_matrix = np.array(phi_d_plus + phi_d_minus)

        # console output
        print("\n" + "=" * 90)
        self.print_duration("Dauer Berechnung Beta 3d-Matrix", (time.time() - start_time) * 1000)

    def _sweep_ride_height(self, steer):
        """
        Solve the position level over the whole ride height range for one steering
        position. Both sweep directions start at z = 0 and every step uses the
        converged solution of the previous one as start vector.
        """
        self.apply_steering(steer)

        beta_plus, beta_minus = [], []
        phi_d_plus, phi_d_minus = [], []

        for z in np.arange(0, self.z_max + self.z_step, self.z_step):
            self._solve_position(z)
            beta_plus.append(np.concatenate((self.beta_start, [z])))
            phi_d_plus.append(self.phi_d_start)
            # keep the solution at z = 0 as start vector of the next steering position
            if z == 0:
                self.beta_start_z0 = self.beta_start
                self.phi_d_start_z0 = self.phi_d_start

        # restart the negative direction from the solution at z = 0
        self.beta_start = self.beta_start_z0
        self.phi_d_start = self.phi_d_start_z0

        for z in np.arange(-self.z_step, self.z_min - self.z_step, -self.z_step):
            self._solve_position(z)
            beta_minus.append(np.concatenate((self.beta_start, [z])))
            phi_d_minus.append(self.phi_d_start)

        # start the next steering position from the solution at z = 0 again
        self.beta_start = self.beta_start_z0
        self.phi_d_start = self.phi_d_start_z0

        beta_plus.reverse()
        phi_d_plus.reverse()
        self.beta_list = np.array(beta_plus + beta_minus)
        self.phi_d_list = np.array(phi_d_plus + phi_d_minus)

    def _solve_position(self, z):
        """
        Solve the position level of the wheel carrier and of the bell crank for one
        ride height and keep both results as start values of the next step.
        """
        self.kine.solve_position(self.beta_start, z)
        self.beta_start = self.kine.beta
        self.kine.calc_position(z, self.tire_radius)

        if self.has_bell_crank:
            self.kine.solve_bell_crank(self.phi_d_start, self.u_d)
            self.phi_d_start = self.kine.phi_d



    # =========================================================================
    # BLOCK 11: CURRENT POSITION AND CHARACTERISTICS
    # =========================================================================

    def live_movement(self, z_curr, steer_curr):
        """
        Characteristics over the whole ride height range on the fine grid for the
        current steering position, followed by the suspension position of the
        sliders that is shown in the 3d view.
        """
        start_time = time.time()

        # the fine grid gives much smoother curves than the coarse start value grid
        z_step_fine = self.z_step / self.FINE_FACTOR
        self.z_array_fine = np.flip(np.arange(self.z_min, self.z_max + z_step_fine, z_step_fine))

        self.apply_steering(steer_curr)
        omega_a, v_a, omegaDot_a, vDot_a = self.steering_velocities()

        # start vectors of the coarse matrix that are closest to the current position
        self.steer_index = self.nearest_index(self.steer_array, steer_curr)
        self.z0_index = self.nearest_index(self.z_array, 0)
        beta_start, phi_d_start = self._start_values(self.steer_index, self.z0_index)

        data_plus, data_minus = self.new_result_buffers(self.results)

        # sweep from the design position upwards, afterwards downwards
        for z in np.arange(0, self.z_max + z_step_fine, z_step_fine):
            beta_start, phi_d_start = self._collect_position(z, beta_start, phi_d_start,
                                                             omega_a, v_a, data_plus)
            if z == 0:
                beta_start_z0, phi_d_start_z0 = beta_start, phi_d_start

        beta_start, phi_d_start = beta_start_z0, phi_d_start_z0

        for z in np.arange(-z_step_fine, self.z_min - z_step_fine, -z_step_fine):
            beta_start, phi_d_start = self._collect_position(z, beta_start, phi_d_start,
                                                             omega_a, v_a, data_minus)

        self.store_sweep_results(self.results, data_plus, data_minus)

        # position of the sliders, this is the one shown in the 3d view
        self.z_index = self.nearest_index(self.z_array, z_curr)
        beta_start, phi_d_start = self._start_values(self.steer_index, self.z_index)
        self.kine.calc_geometry(z_curr, self.zDot, beta_start, phi_d_start, omega_a, v_a,
                                self.tire_radius, self.u_d, self.has_bell_crank, self.F_w, self.F_t)
        self.calc_display_geometry()

        # velocity and acceleration level of the current position
        self.kine.calc_eta(self.zDot, omega_a, v_a, self.kine.MODE_FULL)
        self.kine.calc_etaDot(self.zDot, self.z2Dot, omega_a, v_a, omegaDot_a, vDot_a)

        # console output
        self.print_status(beta_start, (time.time() - start_time) * 1000)

    def live_movement_steer(self, z_curr):
        """
        Characteristics over the steering input on the fine grid at the current
        ride height, counterpart of live_movement() for the steering input. Only
        evaluated for axle types whose steering input is an angle. All results are
        stored in results_over_steer, the x axis of the curves in steer_array_fine.
        """
        if not self.sweeps_steering:
            return

        steer_step_fine = self.steer_step / self.FINE_FACTOR
        omega_a, v_a = self.steering_velocities()[:2]

        # start vectors of the coarse matrix at the straight ahead position
        self.steer0_index = self.nearest_index(self.steer_array, 0)
        self.z_index = self.nearest_index(self.z_array, z_curr)
        beta_start, phi_d_start = self._start_values(self.steer0_index, self.z_index)

        data_plus, data_minus = self.new_result_buffers(self.results_over_steer)

        # sweep from the straight ahead position to the positive steering end,
        # afterwards to the negative steering end
        for steer in np.arange(0, self.steer_max + steer_step_fine, steer_step_fine):
            beta_start, phi_d_start = self._collect_steer_position(steer, z_curr, beta_start,
                                                                   phi_d_start, omega_a, v_a,
                                                                   data_plus)
            if steer == 0:
                beta_start_steer0, phi_d_start_steer0 = beta_start, phi_d_start

        beta_start, phi_d_start = beta_start_steer0, phi_d_start_steer0

        for steer in np.arange(-steer_step_fine, self.steer_min - steer_step_fine, -steer_step_fine):
            beta_start, phi_d_start = self._collect_steer_position(steer, z_curr, beta_start,
                                                                   phi_d_start, omega_a, v_a,
                                                                   data_minus)

        self.store_sweep_results(self.results_over_steer, data_plus, data_minus)

        # x axis of the curves, built from exactly the evaluated steering values,
        # so that x and y data always have the same length
        self.steer_array_fine = self.results_over_steer["steer"]

    def _start_values(self, steer_index, z_index):
        """Start vectors of one point of the coarse start value matrix."""
        beta_start = self.beta_matrix[steer_index, z_index, :5]
        phi_d_start = self.phi_d_matrix[steer_index, z_index]
        return beta_start, phi_d_start

    def _collect_position(self, z, beta_start, phi_d_start, omega_a, v_a, buffer):
        """
        Evaluate one ride height completely and append all characteristics to the
        buffer. Returns the converged values as start vectors of the next step.
        """
        self.kine.calc_geometry(z, self.zDot, beta_start, phi_d_start, omega_a, v_a,
                                self.tire_radius, self.u_d, self.has_bell_crank, self.F_w, self.F_t)
        self.calc_extra_characteristics(omega_a, v_a)

        for key, value in self.extract_results(z).items():
            buffer[key].append(value)

        if self.has_bell_crank:
            phi_d_start = self.kine.phi_d
        return self.kine.beta, phi_d_start

    def _collect_steer_position(self, steer, z, beta_start, phi_d_start, omega_a, v_a, buffer):
        """
        Evaluate one steering position at the ride height z and append all
        characteristics over the steering input to the buffer. Returns the
        converged values as start vectors of the next step.
        """
        self.apply_steering(steer)
        self.kine.calc_steering_plots(z, self.steerDot, beta_start, phi_d_start, omega_a, v_a,
                                      self.tire_radius, self.u_d, self.has_bell_crank)

        for key, value in self.extract_steer_results(steer, z).items():
            buffer[key].append(value)

        if self.has_bell_crank:
            phi_d_start = self.kine.phi_d
        return self.kine.beta, phi_d_start



    # =========================================================================
    # BLOCK 12: CONSOLE OUTPUT
    # =========================================================================

    @staticmethod
    def print_duration(label, duration_ms):
        """One timing line of the console output."""
        print(f"{label:<31} | {duration_ms:>8.3f} ms")

    def print_status(self, beta_start, duration_ms):
        """Print position, velocities, link forces and instant centers to the console."""
        def fmt_arr4(arr):
            return "[" + ", ".join([f"{x:8.4e}" for x in arr]) + "]"
        def fmt_arr2(arr):
            return "[" + ", ".join([f"{x:8.2e}" for x in arr]) + "]"

        print("\n" + "=" * 90)
        print(f"{'KOMPONENTE':<17} | {'WERT'}")
        print("-" * 90)
        print(f"{'Beta Start':<17} | {fmt_arr4(beta_start)}")
        print(f"{'Beta Val':<17} | {fmt_arr4(self.kine.beta)}")
        print(f"{'Eta Val':<17} | {fmt_arr4(self.kine.eta)}")
        print(f"{'EtaDot Val':<17} | {fmt_arr4(self.kine.etaDot)}")
        print(f"{'Lenkerkraefte':<17} | {fmt_arr2(self.kine.F_i)}")
        if self.has_bell_crank:
            print(f"{'F_bearing':<17} | {fmt_arr2(self.kine.F_bearing)}")
            print(f"{'M_bearing':<17} | {fmt_arr2(self.kine.M_bearing)}")
        if not self.kine.lic_pure_translation:
            print(f"{'Laengspol':<17} | {fmt_arr4(self.kine.r_lw)}")
        else:
            print(f"{'Laengspol':<17} | reine Translation, Radius unendlich")
        if not self.kine.rc_pure_translation:
            print(f"{'Querpol':<17} | {fmt_arr4(self.kine.r_tw)}")
            print(f"{'Rollzentrum':<17} | {self.kine.h_RC:>8.2f}")
        else:
            print(f"{'Querpol':<17} | reine Translation, Radius unendlich")
        print("-" * 90)
        self.print_duration("Dauer Berechnung", duration_ms)



    # =========================================================================
    # BLOCK 13: SETUP OF BOTH FIGURES
    # =========================================================================

    def setup_figures(self):
        """
        Create both windows and the fixed axis limits of the 3d view. The limits
        are derived from the design position, so that the model does not jump
        around while the sliders are moved.
        """
        start_time = time.time()

        plt.ion()  # interactive mode for live updates

        # window 1: 3d view of the suspension
        self.fig_3d = plt.figure(num="3D Suspension Kinematics", figsize=(8, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        # window 2: 2d characteristic curves
        self.fig_2d = plt.figure(num="Kinematic Curves (2D)", figsize=(12, 9))
        grid = gridspec.GridSpec(self.PLOT_ROWS, self.PLOT_COLS)
        self.axes_2d = []
        for i in range(len(self.PLOT_CONFIGS)):
            ax = self.fig_2d.add_subplot(grid[i // self.PLOT_COLS, i % self.PLOT_COLS])
            ax.set_xlabel("z [mm]")
            ax.grid(True)
            self.axes_2d.append(ax)

        # optimize the layout, the space at the bottom is left free for the sliders
        self.fig_3d.tight_layout()

        self._calc_axis_limits()

        self.print_duration("Dauer Plot Setup", (time.time() - start_time) * 1000)
        print("=" * 90 + "\n")

    def _calc_axis_limits(self):
        """Fixed limits of the 3d view, large enough for all hardpoints and the tire."""
        P_a, P_b, P_d = self.P_aP0, self.P_bP0, self.P_dP0
        r_bP0, tire_radius = self.r_bP0, self.tire_radius

        x_max = max(max(P_a[:, 0].max(), P_b[:, 0].max(), P_d[:, 0].max()) + 100,
                    r_bP0[0] + tire_radius + 100)
        x_min = min(min(P_a[:, 0].min(), P_b[:, 0].min(), P_d[:, 0].min()) - 100,
                    r_bP0[0] - tire_radius - 100)
        y_max = max(r_bP0[1] + 100, r_bP0[1] + tire_radius + 50)
        y_min = min(P_a[:, 1].min(), P_d[:, 1].min()) - 100
        z_max = max(max(P_a[:5, 2].max(), P_b[:, 2].max(), r_bP0[2]) + 200,
                    P_a[5, 2] + 100, P_d[:, 2].max() + 100, r_bP0[2] + tire_radius + 50)
        z_min = min(min(P_a[:, 2].min(), P_b[:, 2].min(), r_bP0[2] - tire_radius) - 200,
                    P_d[:, 2].min() - 100, r_bP0[2] - tire_radius - 50)

        self.xlim = (x_min, x_max)
        self.ylim = (y_min, y_max)
        self.zlim = (z_min, z_max)



    # =========================================================================
    # BLOCK 14: DRAWING PRIMITIVES FOR THE 3D VIEW
    # =========================================================================

    def _draw_sphere(self, center, radius, color='black'):
        """Draw a joint as a sphere."""
        self.ax_3d.plot_surface(
            center[0] + radius * self._sphere_x,
            center[1] + radius * self._sphere_y,
            center[2] + radius * self._sphere_z,
            color=color, edgecolor='none', shade=True
            )

    def _draw_cylinder(self, P_start, P_end, diameter, color='darkgray'):
        """Draw a link, a rod or a shaft as a cylinder between two points."""
        radius = diameter / 2.0
        axis = np.asarray(P_end, dtype=float) - np.asarray(P_start, dtype=float)
        length = np.linalg.norm(axis)
        if length < 1e-5:
            return
        axis_hat = axis / length

        # any direction that is not parallel to the cylinder axis, used to build
        # the two perpendicular directions u1 and u2 of the cross section
        ref = np.array([1.0, 0.0, 0.0]) if np.abs(axis_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1 = np.cross(axis_hat, ref)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(axis_hat, u1)

        L_GRID, THETA_GRID = np.meshgrid(np.linspace(0, length, 2), np.linspace(0, 2*np.pi, 12))
        X = P_start[0] + L_GRID*axis_hat[0] + radius*np.cos(THETA_GRID)*u1[0] + radius*np.sin(THETA_GRID)*u2[0]
        Y = P_start[1] + L_GRID*axis_hat[1] + radius*np.cos(THETA_GRID)*u1[1] + radius*np.sin(THETA_GRID)*u2[1]
        Z = P_start[2] + L_GRID*axis_hat[2] + radius*np.cos(THETA_GRID)*u1[2] + radius*np.sin(THETA_GRID)*u2[2]

        self.ax_3d.plot_surface(X, Y, Z, color=color, edgecolor='none', shade=True, alpha=0.9)

    def _joint_radius(self, index):
        """Radius of a joint sphere, scaled with the diameter of the connected link."""
        if index < len(self.link_diameters):
            return (self.link_diameters[index] / 2.0) * 1.25
        return 10.0



    # =========================================================================
    # BLOCK 15: UPDATE OF BOTH FIGURES
    # =========================================================================

    def update_figures(self):
        """Redraw the 3d view and all 2d characteristic curves."""
        start_time = time.time()

        self._reset_3d_axes()
        self._draw_links()
        self._draw_spring_damper()
        self._draw_chassis_joints()
        self._draw_wheel_carrier()
        self._draw_steering_axis()
        self.draw_custom_geometry()
        self._draw_curves()
        self._draw_wheel()

        self.ax_3d.set_aspect('equal')
        for fig in (self.fig_3d, self.fig_2d):
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        self.print_duration("Dauer Plot Update", (time.time() - start_time) * 1000)
        print("=" * 90 + "\n")

    def _reset_3d_axes(self):
        """Clear the 3d axes but keep the view angle, cla() would reset it."""
        curr_elev = self.ax_3d.elev if self.ax_3d.elev is not None else 30
        curr_azim = self.ax_3d.azim if self.ax_3d.azim is not None else -60

        self.ax_3d.cla()

        self.ax_3d.view_init(elev=curr_elev, azim=curr_azim)
        self.ax_3d.set_proj_type('ortho')  # orthographic / isometric view
        self.ax_3d.set_xlim(self.xlim)
        self.ax_3d.set_ylim(self.ylim)
        self.ax_3d.set_zlim(self.zlim)
        self.ax_3d.set_xlabel('X [mm]')
        self.ax_3d.set_ylabel('Y [mm]')
        self.ax_3d.set_zlabel('Z [mm]')

    def _draw_links(self):
        """The five links as thick tubes between chassis and wheel carrier."""
        for i in range(5):
            self._draw_cylinder(self.kine.P_a[i], self.kine.P_b[i],
                                self.link_diameters[i], color='darkgray')

    def _draw_spring_damper(self):
        """
        Spring/damper element, either mounted directly between wheel carrier and
        chassis or driven by a push-/pullrod and a bell crank.
        """
        if self.has_bell_crank:
            # push-/pullrod (rod 5) between wheel carrier and bell crank
            self._draw_cylinder(self.kine.P_b[5], self.kine.P_d[0],
                                self.link_diameters[5], color='darkgray')
            # spring/damper element between bell crank and chassis
            self._draw_cylinder(self.kine.P_d[1], self.kine.P_a[5], 10, color='purple')

            # bell crank: both arms and the connection of its hardpoints
            for i in range(2):
                self._draw_cylinder(self.kine.P_d[i], self.kine.r_d, 10, color='brown')
            self._draw_cylinder(self.kine.P_d[0], self.kine.P_d[1], 10, color='brown')

            # joints of the bell crank, of the spring/damper and of the pivot
            self._draw_sphere(self.kine.P_d[0], self._joint_radius(5), color='darkgray')
            self._draw_sphere(self.kine.P_d[1], 10, color='black')
            self._draw_sphere(self.kine.r_d, 10, color='black')
            self._draw_sphere(self.kine.P_a[5], 10, color='black')

            # wheel carrier joints incl. the push-/pullrod joint
            for i, point in enumerate(self.kine.P_b[:6]):
                self._draw_sphere(point, self._joint_radius(i), color='darkgray')
        else:
            # direct configuration: coilover between wheel carrier and chassis
            self._draw_cylinder(self.kine.P_b[5], self.kine.P_a[5],
                                self.link_diameters[5], color='purple')

            # wheel carrier joints of the five links
            for i, point in enumerate(self.kine.P_b[:5]):
                self._draw_sphere(point, self._joint_radius(i), color='darkgray')

            # both mounting points of the coilover
            self._draw_sphere(self.kine.P_b[5], self._joint_radius(5), color='purple')
            self._draw_sphere(self.kine.P_a[5], self._joint_radius(5), color='purple')

    def _draw_chassis_joints(self):
        """Joints of the five links on the chassis side."""
        for i, point in enumerate(self.kine.P_a[:5]):
            self._draw_sphere(point, self._joint_radius(i), color='darkgray')

    def _draw_wheel_carrier(self):
        """
        Wire frame of the wheel carrier: all hardpoints are connected to the outer
        point of the wheel axis P_b,6. Additionally the wheel axis itself and the
        connection from the wheel center to the wheel contact patch.
        """
        P_b6 = self.kine.P_b[6]

        # structure of the wheel carrier (blue)
        for i in range(6):
            self._draw_cylinder(self.kine.P_b[i], P_b6, 10, color='blue')

        # wheel rotation axis (green) and wheel center
        self._draw_cylinder(self.kine.r_b, P_b6, 10, color='green')
        self._draw_sphere(self.kine.r_b, 10, color='green')

        # connection to the wheel contact patch (red)
        contact_patch = self.kine.r_b + self.kine.r_wb
        self._draw_cylinder(self.kine.r_b, contact_patch, 10, color='red')
        self._draw_sphere(contact_patch, 10, color='red')

    def _draw_steering_axis(self):
        """Approximated instantaneous steering axis (cyan)."""
        self._draw_cylinder(self.kine.rot_axis_low, self.kine.rot_axis_high, 10, color='cyan')

    def _draw_wheel(self):
        """
        Tire as a semi transparent tread surface around the wheel rotation axis,
        with both outer edges drawn sharply.
        """
        r_b = self.kine.r_b        # wheel center
        P_b6 = self.kine.P_b[6]    # outer point of the wheel rotation axis
        r_wb = self.kine.r_wb      # vector from the wheel center to the contact patch

        # normal of the wheel plane (along the wheel rotation axis)
        n_hat = (P_b6 - r_b) / np.linalg.norm(P_b6 - r_b)
        # first direction in the wheel plane (pointing down to the ground)
        u1_hat = r_wb / np.linalg.norm(r_wb)
        # second direction in the wheel plane (perpendicular to the other two)
        u2_hat = np.cross(n_hat, u1_hat)
        u2_hat = u2_hat / np.linalg.norm(u2_hat)

        # grid of the tread surface
        theta = np.linspace(0, 2 * np.pi, 60)
        width = np.linspace(-self.tire_width / 2, self.tire_width / 2, 2)
        THETA, WIDTH = np.meshgrid(theta, width)

        X = r_b[0] + self.tire_radius * (np.cos(THETA)*u1_hat[0] + np.sin(THETA)*u2_hat[0]) + WIDTH*n_hat[0]
        Y = r_b[1] + self.tire_radius * (np.cos(THETA)*u1_hat[1] + np.sin(THETA)*u2_hat[1]) + WIDTH*n_hat[1]
        Z = r_b[2] + self.tire_radius * (np.cos(THETA)*u1_hat[2] + np.sin(THETA)*u2_hat[2]) + WIDTH*n_hat[2]
        self.ax_3d.plot_surface(X, Y, Z, color='black', alpha=0.25, shade=True)

        # both outer edges of the tire
        for offset in (-self.tire_width / 2, self.tire_width / 2):
            X_edge = r_b[0] + self.tire_radius * (np.cos(theta)*u1_hat[0] + np.sin(theta)*u2_hat[0]) + offset*n_hat[0]
            Y_edge = r_b[1] + self.tire_radius * (np.cos(theta)*u1_hat[1] + np.sin(theta)*u2_hat[1]) + offset*n_hat[1]
            Z_edge = r_b[2] + self.tire_radius * (np.cos(theta)*u1_hat[2] + np.sin(theta)*u2_hat[2]) + offset*n_hat[2]
            self.ax_3d.plot(X_edge, Y_edge, Z_edge, color='black', linewidth=1.5)

    def _curve_data(self, key, x_axis):
        """
        x values, y values and x label of one characteristic curve. Curves over
        the steering input are read from results_over_steer and stay empty if the
        subclass does not sweep the steering angle (both data are None then).
        """
        if x_axis == "steer":
            return self.steer_array_fine, self.results_over_steer[key], "steer [°]"
        return self.z_array_fine, self.results[key], "z [mm]"

    def _draw_curves(self):
        """Redraw all 2d characteristic curves."""
        for ax, (key, unit, x_axis) in zip(self.axes_2d, self.PLOT_CONFIGS):
            ax.cla()
            ax.set_ylabel(f"{key} [{unit}]")
            ax.grid(True)

            x_data, y_data, x_label = self._curve_data(key, x_axis)
            ax.set_xlabel(x_label)
            if x_data is None or y_data is None:
                continue

            ax.plot(x_data, y_data, color='blue', label=key)
            ax.legend(loc='upper right', fontsize='small')



    # =========================================================================
    # BLOCK 16: INTERACTIVE CONTROLS
    # =========================================================================

    def add_controls(self):
        """
        Turn both figures into an interactive plot: two synchronized sliders for
        ride height and steering input, buttons for the standard views and a
        shortcut to close both windows.
        """
        self._adjust_layout_for_controls()
        self._create_sliders()

        # recalculation and redraw, only triggered when the mouse button is released
        def recalculate_and_redraw(event=None):
            # both sliders are synchronized, so the 3d values are representative
            z_val = self.z_slider_3d.val
            steer_val = self.steer_slider_3d.val

            # only update if the values really changed
            if event is not None and self.z_curr == z_val and self.steer_curr == steer_val:
                return

            self.z_curr = z_val
            self.steer_curr = steer_val

            self.live_movement_steer(z_val)
            self.live_movement(z_val, steer_val)
            self.update_figures()

        self._connect_events(recalculate_and_redraw)
        self._create_view_buttons()

        # initial call at start up
        recalculate_and_redraw(None)

    def _adjust_layout_for_controls(self):
        """Leave space at the bottom of both windows for the sliders."""
        self.fig_3d.subplots_adjust(bottom=0.15)
        self.fig_2d.subplots_adjust(
            bottom=0.15,   # space for the sliders
            top=0.96,      # move the upper row closer to the window border
            left=0.04,     # use the empty space on the left
            right=0.98,    # use the empty space on the right
            wspace=0.35,   # horizontal distance between the plots
            hspace=0.40    # vertical distance between the plots
            )

    def _create_sliders(self):
        """One pair of sliders per window, kept synchronized while dragging."""
        ax_z_3d = self.fig_3d.add_axes([0.2, 0.07, 0.6, 0.02])
        ax_steer_3d = self.fig_3d.add_axes([0.2, 0.02, 0.6, 0.02])
        self.z_slider_3d = Slider(ax_z_3d, 'z_curr [mm]', self.z_min, self.z_max, valinit=0.0, valfmt='%1.4f')
        self.steer_slider_3d = Slider(ax_steer_3d, 'steer_curr', self.steer_min, self.steer_max, valinit=0.0, valfmt='%1.4f')

        ax_z_2d = self.fig_2d.add_axes([0.2, 0.07, 0.6, 0.02])
        ax_steer_2d = self.fig_2d.add_axes([0.2, 0.02, 0.6, 0.02])
        self.z_slider_2d = Slider(ax_z_2d, 'z_curr [mm]', self.z_min, self.z_max, valinit=0.0, valfmt='%1.4f')
        self.steer_slider_2d = Slider(ax_steer_2d, 'steer_curr', self.steer_min, self.steer_max, valinit=0.0, valfmt='%1.4f')

        self.z_curr = self.z_slider_3d.val
        self.steer_curr = self.steer_slider_3d.val

        # mirror every slider to its counterpart in the other window, the events
        # of the mirrored slider are switched off to avoid an endless loop
        def mirror(source, target):
            def on_changed(val):
                target.eventson = False
                target.set_val(val)
                target.eventson = True
            source.on_changed(on_changed)

        mirror(self.z_slider_3d, self.z_slider_2d)
        mirror(self.z_slider_2d, self.z_slider_3d)
        mirror(self.steer_slider_3d, self.steer_slider_2d)
        mirror(self.steer_slider_2d, self.steer_slider_3d)

    def _connect_events(self, on_release):
        """Recalculate on mouse release and close both windows on 'q' or 'escape'."""
        def on_key_press(event):
            if event.key in ['q', 'escape']:
                plt.close(self.fig_3d)
                plt.close(self.fig_2d)

        for fig in (self.fig_3d, self.fig_2d):
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('key_press_event', on_key_press)

    def _create_view_buttons(self):
        """Buttons next to the sliders that switch the 3d view to a standard plane."""
        # (text, elevation, azimuth, x position, y position)
        button_config = [
            ("XY-Ebene", 90, 90, 0.85, 0.07),
            ("XZ-Ebene", 0, 90, 0.91, 0.07),
            ("YZ-Ebene (X+)", 0, 0, 0.85, 0.02),
            ("YZ-Ebene (X-)", 0, 180, 0.91, 0.02)
            ]

        def make_callback(elev, azim):
            def on_click(event):
                self.ax_3d.view_init(elev=elev, azim=azim)
                self.fig_3d.canvas.draw_idle()
            return on_click

        self.view_button_axes = []
        self.view_buttons = []
        for text, elev, azim, x, y in button_config:
            ax_button = self.fig_3d.add_axes([x, y, 0.05, 0.035])  # [left, bottom, width, height]
            button = Button(ax_button, text)
            button.on_clicked(make_callback(elev, azim))

            # keep the references, otherwise matplotlib drops the buttons
            self.view_button_axes.append(ax_button)
            self.view_buttons.append(button)


# =============================================================================
# BLOCK 17: FIVE-LINK AXLE WITH A TRANSLATIONAL STEERING INPUT
# =============================================================================

class LiveMovementStd5Link(LiveMovementBase):
    """
    Five-link axle with a conventional steering input.

    The chassis hardpoints marked by a_i are shifted translationally in y
    direction, they are not rotated. Therefore body a only has a translational
    velocity v_a, while the rotation angles phi_a, the rotation axes u_a and the
    angular velocity omega_a stay zero.
    """

    # no inner knuckle, so the inner knuckle loads and the steering ratio do not exist
    has_rotating_knuckle = False

    # the steering input is a translation in mm, no curves over the steering input
    sweeps_steering = False

    def __init__(self, kine_object):
        super().__init__(kine_object)

        # constant zero quantities of body a for this axle type
        self.phi_a = np.zeros(5)          # no rotation of the chassis hardpoints
        self.u_a = np.zeros((5, 3))       # therefore also no rotation axes
        self.r_aP0 = np.zeros((5, 3))     # the hardpoints are their own frame origin
        self.omega_a = np.zeros((5, 3))   # no angular velocity of body a

    def set_axle_data(self, P_aP0, P_bP0, P_dP0, r_bP0, r_dP0, a_i, u_d, has_bell_crank,
                      link_diameters, tire_radius, tire_width):
        """
        Geometry and hardware of the axle, see _store_axle_data() of the base
        class. This axle type needs neither r_aP0 nor u_a, both stay zero.
        """
        self._store_axle_data(P_aP0, P_bP0, P_dP0, r_bP0, r_dP0, a_i, u_d, has_bell_crank,
                              link_diameters, tire_radius, tire_width)

    def apply_steering(self, steer):
        """
        Translational steering input: shift the selected chassis hardpoints by
        r_aDelta in y direction and hand the geometry over to the kinematics.
        """
        r_aDelta = self.a_i[:, np.newaxis] * np.array([0.0, steer, 0.0])
        self.kine.set_geometry(self.P_aP0, self.P_bP0, self.P_dP0, self.r_aP0, r_aDelta,
                               self.r_bP0, self.r_dP0, self.phi_a, self.u_a)

    def steering_velocities(self):
        """
        The chassis hardpoints only translate, so the prescribed steering velocity
        appears in v_a and the steering acceleration in vDot_a, while omega_a and
        omegaDot_a stay zero.
        """
        direction = self.a_i[:, np.newaxis] * np.array([0.0, 1.0, 0.0])
        v_a = direction * self.steerDot
        vDot_a = direction * self.steer2Dot
        return self.omega_a, v_a, self.omega_a, vDot_a





# =============================================================================
# BLOCK 18: FIVE-LINK AXLE WITH A ROTATING INNER KNUCKLE
# =============================================================================

class LiveMovementRotationAxis5Link(LiveMovementBase):
    """
    Five-link axle with the unconventional steering input.

    The chassis hardpoints marked by a_i rotate by the steering angle phi_a about
    the axis u_a of the inner knuckle, they are not shifted. Therefore body a only
    has an angular velocity omega_a, while the translational offset r_aDelta and
    the velocity v_a stay zero.
    """

    # the chassis hardpoints rotate about the axis of the inner knuckle
    has_rotating_knuckle = True

    # the steering input is an angle in rad, so the curves over the steering input exist
    sweeps_steering = True

    def __init__(self, kine_object):
        super().__init__(kine_object)

        # constant zero quantities of body a for this axle type
        self.r_aDelta = np.zeros((5, 3))  # no translation of the chassis hardpoints
        self.v_a = np.zeros((5, 3))       # therefore also no translational velocity

    def set_axle_data(self, P_aP0, P_bP0, P_dP0, r_aP0, r_bP0, r_dP0, a_i, u_a, u_d,
                      has_bell_crank, link_diameters, bearing_sizes, tire_radius, tire_width):
        """
        Geometry and hardware of the axle, see _store_axle_data() of the base class.
        In addition this axle type needs the frame origins r_aP0 and the rotation
        axes u_a of the inner knuckle as well as the bearing sizes on that axis.
        """
        self._store_axle_data(P_aP0, P_bP0, P_dP0, r_bP0, r_dP0, a_i, u_d, has_bell_crank,
                              link_diameters, tire_radius, tire_width, bearing_sizes)
        self.r_aP0 = np.asarray(r_aP0, dtype=float)
        self.u_a = np.asarray(u_a, dtype=float)

    def apply_steering(self, steer):
        """
        Rotational steering input: rotate the selected chassis hardpoints by the
        angle phi_a about u_a and hand the geometry over to the kinematics.
        """
        phi_a = self.a_i[:, np.newaxis] * steer
        self.kine.set_geometry(self.P_aP0, self.P_bP0, self.P_dP0, self.r_aP0, self.r_aDelta,
                               self.r_bP0, self.r_dP0, phi_a, self.u_a)

    def steering_velocities(self):
        """
        The chassis hardpoints only rotate, so the prescribed steering velocity
        appears in omega_a = steerDot * u_a and the steering acceleration in
        omegaDot_a, while v_a and vDot_a stay zero.
        """
        omega_direction = self.u_a * self.a_i[:, np.newaxis]
        omega_a = omega_direction * self.steerDot
        omegaDot_a = omega_direction * self.steer2Dot
        return omega_a, self.v_a, omegaDot_a, self.v_a

    def calc_extra_characteristics(self, omega_a, v_a):
        """
        Forces and moments at the inner knuckle and the steering ratio, both only
        exist for this axle type.
        """
        self.kine.calc_inner_knuckle_loads(self.u_a, self.a_i, self.F_w)
        self.kine.calc_steering_ratio(omega_a, v_a, self.steerDot)

    def calc_display_geometry(self):
        """Rotation axis of the inner knuckle, only needed for the 3d view."""
        self.kine.calc_inner_axis_display(self.u_a, self.r_aP0, self.P_aP0)

    def draw_custom_geometry(self):
        """Rotation axis of the inner knuckle and the bearings sitting on it."""
        if self.kine.P_axis_a_upper is None or self.kine.P_axis_a_lower is None:
            return

        P_upper = self.kine.P_axis_a_upper
        P_lower = self.kine.P_axis_a_lower

        # the rotation axis itself
        self._draw_cylinder(P_upper, P_lower, diameter=10, color='cyan')

        # the bearings are given by their z range, the corresponding points on the
        # rotation axis are found by interpolating between its end points
        dz = P_lower[2] - P_upper[2]
        for z_start, z_end, diameter in self.bearing_sizes:
            t_start = (z_start - P_upper[2]) / dz
            t_end = (z_end - P_upper[2]) / dz
            P_bearing_start = P_upper + t_start * (P_lower - P_upper)
            P_bearing_end = P_upper + t_end * (P_lower - P_upper)
            self._draw_cylinder(P_bearing_start, P_bearing_end, diameter=diameter, color='orange')











# =============================================================================
# BLOCK 19: MAIN PROGRAM
# =============================================================================

def build_std_simulation(kine_object):
    """Interactive simulation of the conventional five-link axle (STD)."""
    simulation = LiveMovementStd5Link(kine_object)
    simulation.set_axle_data(P_aP0_STD, P_bP0_STD, P_dP0_STD, r_bP0_STD, r_dP0_STD,
                             a_i_STD, u_d_STD, has_bell_crank_STD, link_diameters_STD,
                             tire_radius, tire_width)
    simulation.set_simulation_data(z_min, z_max, z_step, steer_min_STD, steer_max_STD,
                                   steer_step_STD, zDot, z2Dot, steerDot_STD, steer2Dot_STD,
                                   F_w, F_t)
    return simulation


def build_rot_simulation(kine_object):
    """Interactive simulation of the five-link axle with a rotating inner knuckle (ROT)."""
    simulation = LiveMovementRotationAxis5Link(kine_object)
    simulation.set_axle_data(P_aP0_ROT, P_bP0_ROT, P_dP0_ROT, r_aP0_ROT, r_bP0_ROT, r_dP0_ROT,
                             a_i_ROT, u_a_ROT, u_d_ROT, has_bell_crank_ROT, link_diameters_ROT,
                             bearing_sizes_ROT, tire_radius, tire_width)
    simulation.set_simulation_data(z_min, z_max, z_step, steer_min_ROT, steer_max_ROT,
                                   steer_step_ROT, zDot, z2Dot, steerDot_ROT, steer2Dot_ROT,
                                   F_w, F_t)
    return simulation


# Both simulations share one kinematics object and are built at import time, so
# that they are available in an interactive console as well. Only the one that is
# selected by axle_config in BLOCK 0 is actually solved and plotted.
kine = Kinematics()
simulation_STD = build_std_simulation(kine)
simulation_ROT = build_rot_simulation(kine)

simulations = {"std": simulation_STD, "rot": simulation_ROT}


if __name__ == "__main__":

    simulation = simulations[axle_config]

    simulation.calc_start_values()   # coarse start value matrix over z and steering
    simulation.setup_figures()       # create the 3d window and the 2d window
    simulation.add_controls()        # sliders, buttons and the first update

    plt.show(block=True)