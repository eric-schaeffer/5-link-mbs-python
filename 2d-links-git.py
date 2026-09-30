#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinematics of the steerable five-link axle, reduced to a planar multibody system.

Structure of this file:
    BLOCK 0      : input data of the simulation
    BLOCK 1 - 6  : Kinematics2d    - planar multibody kinematics of the axle
    BLOCK 7 - 12 : LiveMovement2d  - steering sweep, plan view and slider
    BLOCK 13     : main program

Notation follows the documentation:
    body a : chassis side  (the fixed hardpoint of the trailing link and the
             hardpoints of the transverse links that rotate with the inner knuckle)
    body b : wheel carrier (outer knuckle)
    beta   : primary joint coordinates [theta, x, y]
    eta    : velocity level unknowns   [omega_b, v_bx, v_by]
    g^s    : implicit closure conditions, G^s_a / G^s_b the corresponding
             constraint matrices of body a and body b
    c^perp : orthogonal vector c^perp = [-c_y, c_x]^T, the planar replacement of
             the skew symmetric matrix c~ of the spatial model
    r_M    : instant center (Momentanpol) of the wheel carrier

The plane is the x-y-plane of the spatial model seen from above, therefore the
wheel appears as a rectangle of tyre diameter and tyre width.

All lengths are in mm, all angles in rad unless a plot label says otherwise.

@author: eric
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time


# ==========================================
# BLOCK 0: INPUT DATA OF THE SIMULATION
# ==========================================

# --- steering sweep, the steering input is the rotation angle of the inner knuckle ---

steer_min = -0.85     # lower end of the steering range
steer_max = 0.30      # upper end of the steering range
steer_step = 0.01    # step width of the coarse start value grid and of the slider
steer_init = 0.0      # steering angle the window starts with

steerDot = 3          # prescribed steering velocity omega_a of the inner knuckle


# --- chassis hardpoints in design position (P0) --------------------------------------

P_a0P0 = np.array([-420, 425])    # P1, trailing link, does not follow the steering
P_a1P0 = np.array([240, 300])     # P3
P_a2P0 = np.array([-120, 330])    # P5
P_aP0 = [P_a0P0, P_a1P0, P_a2P0]

# origin of the body fixed frames K_a, identical for all chassis hardpoints because
# the steered ones all rotate about the same pivot of the inner knuckle. For the
# trailing link the same origin is used, its lever arm c_a simply stays constant.
r_a = np.array([60, 340])
r_aP0 = np.tile(r_a, (3, 1))


# --- wheel carrier hardpoints in design position (P0) --------------------------------

r_bP0 = np.array([110, 850])      # wheel centre, origin of the body fixed frame K_b
P_b0P0 = np.array([140, 720])     # P2
P_b1P0 = np.array([240, 730])     # P4
P_b2P0 = np.array([30, 710])      # P6
P_bP0 = [P_b0P0, P_b1P0, P_b2P0]


# --- steered links -------------------------------------------------------------------

# selection vector, marks the links that follow the steering input (here links 1 and 2)
a_i = [0, 1, 1]


# --- tyre 215/60R18, only used for the plan view -------------------------------------

tire_radius = 358
tire_width = 215


class Kinematics2d:
    """
    Planar multibody kinematics of the five-link axle.

    Index convention of the hardpoint arrays:
        P_a[0] / P_b[0]   : trailing link, its chassis hardpoint is fixed
        P_a[1:] / P_b[1:] : transverse links, their chassis hardpoints rotate
                            about the pivot r_a of the inner knuckle

    Typical call order for one steering position:
        set_geometry() -> solve_position() -> calc_position()
        -> calc_Gs_a() -> calc_Gs_b() -> calc_eta() -> calc_instant_center()
    (calc_geometry() performs this sequence in one call.)
    """

    # Tolerance for the degeneracy check (pure translation of the wheel carrier)
    EPS = 1e-12

    def __init__(self):
        # --- geometry, defined by set_geometry() ---------------------------
        self.a_c_a = None        # {}^a c_a : body fixed lever arms, chassis side (3, 2)
        self.b_c_b = None        # {}^b c_b : body fixed lever arms, wheel carrier (3, 2)
        self.l_i_const = None    # constant link lengths l of the three links (3,)
        self.c_a = None          # c_a in the inertial frame K_0 (3, 2)
        self.P_a = None          # chassis side hardpoints (3, 2)

        # --- position level -------------------------------------------------
        self.beta = None         # primary joint coordinates [theta, x, y]
        self.r_b = None          # origin of the wheel carrier frame K_b = [x, y]
        self.T_0b = None         # {}^{0b}T of the current wheel carrier rotation
        self.c_b = None          # c_b in the inertial frame K_0 (3, 2)
        self.P_b = None          # wheel carrier hardpoints (3, 2)
        self.l_i = None          # distance vectors l_i = P_b,i - P_a,i (3, 2)

        # --- velocity level ---------------------------------------------------
        self.Gs_a = None         # constraint matrix G^s_a (3, 3)
        self.Gs_b = None         # constraint matrix G^s_b (3, 3)
        self.eta = None          # velocity level solution [omega_b, v_bx, v_by]
        self.omega_b = None      # angular velocity scalar of the wheel carrier
        self.v_b = None          # velocity of the wheel carrier

        # --- suspension characteristics ----------------------------------------
        self.r_M = None          # instant center (Momentanpol) of the wheel carrier
        self.pure_translation = True
        self.i_steer = None      # steering ratio omega_a / omega_b

    # ==========================================
    # BLOCK 1: ROTATION MATRIX AND CLOSURE CONDITIONS (POSITION LEVEL)
    # ==========================================

    @staticmethod
    def _T_0i(psi):
        """
        Planar rotation matrix about the angle psi. It is used for the chassis side
        ({}^{0a}T(phi)) as well as for the wheel carrier ({}^{0b}T(theta)).
        """
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s],
                         [s, c]])

    @staticmethod
    def _perp(c):
        """
        Orthogonal vector c^perp = [-c_y, c_x]^T, evaluated row wise for an array
        of planar vectors. It replaces the skew symmetric matrix c~ of the spatial
        model, because the time derivative of a rotating planar vector is
        cDot = omega * c^perp.
        """
        return np.column_stack([-c[:, 1], c[:, 0]])

    def _c_b_func(self, theta):
        """c_b = {}^{0b}T(theta) * {}^b c_b for all wheel carrier hardpoints."""
        return (self._T_0i(theta) @ self.b_c_b.T).T

    def _l_i_func(self, beta):
        """
        Distance vectors of the three links,
            l_i = P_b,i - P_a,i = r_b + c_b,i - P_a,i
        evaluated for the primary joint coordinates beta = [theta, x, y].
        """
        theta, x, y = beta
        return np.array([x, y]) + self._c_b_func(theta) - self.P_a

    def _g_s_i(self, beta):
        """
        Implicit closure conditions of the three links on position level,
            g^s = 1/2 * (l^T l - l_const^2) = 0
        Residual vector (3,) handed to least_squares.
        """
        l_i = self._l_i_func(beta)
        return 0.5 * (np.sum(l_i ** 2, axis=1) - self.l_i_const ** 2)

    # ==========================================
    # BLOCK 2: GEOMETRY SETUP (called on every steering update)
    # ==========================================

    def set_geometry(self, P_aP0, P_bP0, r_aP0, r_bP0, phi_a):
        """
        Set up all quantities that only depend on the steering input.

        P_aP0 : chassis side hardpoints in design position P0 (3, 2)
        P_bP0 : wheel carrier hardpoints in design position P0 (3, 2)
        r_aP0 : origins of the body fixed frames K_a in design position (3, 2)
        r_bP0 : origin of the body fixed frame K_b in design position (wheel centre)
        phi_a : rotation angle of each chassis hardpoint about the pivot r_a (3,)
        """
        P_aP0 = np.asarray(P_aP0, dtype=float)
        P_bP0 = np.asarray(P_bP0, dtype=float)
        r_aP0 = np.asarray(r_aP0, dtype=float)
        r_bP0 = np.asarray(r_bP0, dtype=float)
        # phi_a may be handed over as (3,) or (3, 1), flatten it for both cases
        phi_a = np.asarray(phi_a, dtype=float).reshape(-1)

        # body fixed lever arms in the design position:
        #   {}^a c_a = P_a,P0 - r_a,P0 ,  {}^b c_b = P_b,P0 - r_b,P0
        self.a_c_a = P_aP0 - r_aP0
        self.b_c_b = P_bP0 - r_bP0

        # constant link lengths in the design position
        self.l_i_const = np.linalg.norm(P_bP0 - P_aP0, axis=1)

        # rotate the chassis side lever arms into the inertial frame K_0:
        #   c_a = {}^{0a}T * {}^a c_a
        # The trailing link is not steered, its angle phi_a stays zero and its
        # lever arm therefore keeps the constant value P_a,P0 - r_a,P0.
        self.c_a = np.array([self._T_0i(phi_a[i]) @ self.a_c_a[i]
                             for i in range(len(phi_a))])

        # current chassis hardpoints
        self.P_a = r_aP0 + self.c_a

    # ==========================================
    # BLOCK 3: POSITION LEVEL
    # ==========================================

    def solve_position(self, beta_start):
        """
        Solve the nonlinear closure conditions on position level for the primary
        joint coordinates beta (least_squares, Levenberg-Marquardt).
        """
        res = least_squares(self._g_s_i, np.array(beta_start, dtype=float), method='lm')
        if not res.success:
            print(f"Wheel carrier: least_squares warning! Reason: {res.message} "
                  f"Max. residual: {np.max(np.abs(res.fun)):.4e}")
        self.beta = res.x.copy()
        # r_b = [x, y] : position of the wheel carrier frame origin
        self.r_b = self.beta[1:3].copy()

    def calc_position(self):
        """
        Evaluate all position level quantities for the converged beta:
        distance vectors l_i, lever arms c_b and the hardpoints P_b.
        """
        theta = self.beta[0]
        self.T_0b = self._T_0i(theta)
        self.l_i = self._l_i_func(self.beta)
        self.c_b = self._c_b_func(theta)
        self.P_b = self.r_b + self.c_b

    # ==========================================
    # BLOCK 4: VELOCITY LEVEL
    # ==========================================

    def calc_Gs_a(self):
        """
        Constraint matrix of body a (chassis side):
            G^s_a = [ -l^T c_a^perp   -l^T ]
        """
        rot_part = -np.sum(self.l_i * self._perp(self.c_a), axis=1)[:, np.newaxis]
        self.Gs_a = np.hstack([rot_part, -self.l_i])

    def calc_Gs_b(self):
        """
        Constraint matrix of body b (wheel carrier):
            G^s_b = [ l^T c_b^perp   l^T ]
        """
        rot_part = np.sum(self.l_i * self._perp(self.c_b), axis=1)[:, np.newaxis]
        self.Gs_b = np.hstack([rot_part, self.l_i])

    def calc_eta(self, omega_a, v_a):
        """
        Solve the linear closure conditions on velocity level for
        eta = [omega_b, v_bx, v_by]. With gamma^s_ab = 0 the closure conditions
            G^s_a eta_a + G^s_b eta_b = 0
        reduce to G^s_b eta_b = -G^s_a eta_a.

        omega_a : angular velocity scalar of each chassis hardpoint (3,)
        v_a     : translational velocity of each chassis hardpoint (3, 2)
        """
        eta_a = np.column_stack([np.asarray(omega_a, dtype=float),
                                 np.asarray(v_a, dtype=float)])
        Gs_a_eta_a = np.sum(self.Gs_a * eta_a, axis=1)  # row wise scalar product

        self.eta = np.linalg.solve(self.Gs_b, -Gs_a_eta_a)
        self.omega_b = self.eta[0]
        self.v_b = self.eta[1:3]
        return self.eta

    # ==========================================
    # BLOCK 5: SUSPENSION CHARACTERISTICS
    # ==========================================

    def calc_instant_center(self, steerDot):
        """
        Instant center (Momentanpol) of the wheel carrier and steering ratio
        i_steer = omega_a / omega_b, i.e. the ratio of the angular velocities of
        inner and outer knuckle.

        steerDot : prescribed steering velocity omega_a of the inner knuckle
        """
        self.pure_translation = abs(self.omega_b) < self.EPS
        if self.pure_translation:
            # no instant center exists, the wheel carrier translates
            self.r_M = np.array([np.inf, np.inf])
            self.i_steer = np.inf
            return

        # analytic simplification of r_M = r_b + (omega x v) / |omega|^2 in the plane
        self.r_M = self.r_b + np.array([-self.v_b[1] / self.omega_b,
                                        self.v_b[0] / self.omega_b])
        self.i_steer = steerDot / self.omega_b

    # ==========================================
    # BLOCK 6: DRIVER METHOD FOR THE PLOT
    # ==========================================

    def calc_geometry(self, beta_start, omega_a, v_a, steerDot):
        """Complete evaluation of one steering position (plan view and characteristics)."""
        self.solve_position(beta_start)
        self.calc_position()
        self.calc_Gs_a()
        self.calc_Gs_b()
        self.calc_eta(omega_a, v_a)
        self.calc_instant_center(steerDot)


class LiveMovement2d:
    """
    Interactive planar simulation of the five-link axle.

    The class owns the kinematics object, the axle data and the plan view window
    with its steering slider. It also contains the sweep over the steering range
    that supplies the start vectors of the nonlinear solver.

    The steering input is the rotation angle of the inner knuckle: the chassis
    hardpoints marked by a_i rotate about the pivot r_a, they are not shifted.
    Therefore body a only has an angular velocity omega_a, while v_a stays zero.

    Typical call order:
        set_axle_data()        -> geometry and hardware of the axle
        set_simulation_data()  -> steering range and prescribed steering velocity
        calc_start_values()    -> coarse start value matrix over the steering range
        setup_figure()         -> create the window
        add_controls()         -> slider and the first update
    """

    # --- colours, sizes and layout of the plan view ---------------------------
    COLOR_LINK = 'black'             # the three links
    COLOR_CHASSIS = 'blue'           # steered chassis hardpoints and inner knuckle
    COLOR_CHASSIS_FIXED = 'green'    # fixed chassis hardpoint of the trailing link
    COLOR_PIVOT = 'purple'           # pivot of the inner knuckle
    COLOR_WHEEL_CARRIER = 'red'      # wheel carrier hardpoints and body
    COLOR_INSTANT_CENTER = 'cyan'    # instant center of the wheel carrier
    COLOR_TIRE = 'gray'              # tread rectangle of the wheel

    MARKER_SIZE = 120                # size of the hardpoint markers
    LINE_WIDTH = 2.5                 # line width of links, wheel carrier and knuckle
    PLOT_MARGIN = 100                # distance between geometry and axis limits

    # refinement of the slider compared to the coarse start value grid
    FINE_FACTOR = 5

    # console output of every slider step, set it to False for a quiet console
    PRINT_STATUS = True

    def __init__(self, kine_object):
        self.kine = kine_object

        # --- axle data, filled by set_axle_data() -----------------------------
        self.P_aP0 = None              # chassis hardpoints in design position (3, 2)
        self.P_bP0 = None              # wheel carrier hardpoints in design position (3, 2)
        self.r_aP0 = None              # origins of the body fixed frames K_a (3, 2)
        self.r_bP0 = None              # origin of the body fixed frame K_b (wheel centre)
        self.a_i = None                # selection vector, marks the steered links
        self.tire_radius = None
        self.tire_width = None

        # --- simulation data, filled by set_simulation_data() -----------------
        self.steer_min = None          # steering range and step width
        self.steer_max = None
        self.steer_step = None
        self.steer_init = None
        self.steerDot = None           # prescribed steering velocity

        # the chassis hardpoints only rotate, so their translational velocity is zero
        self.v_a = None

        # --- sweep grid and start values of the nonlinear solver ---------------
        self.steer_array = None        # coarse steering grid (start value matrix)
        self.beta_start = None         # start vector of the primary joint coordinates
        self.beta_matrix = None        # beta over the steering range (2d matrix)

        # --- current slider position --------------------------------------------
        self.steer_curr = 0.0
        self.steer_index = 0           # index of the closest coarse steering value

        # --- figure and axes -------------------------------------------------------
        self.fig = None
        self.ax = None
        self.xlim = None               # fixed limits of the plan view
        self.ylim = None
        self.steer_slider = None

    # ==========================================
    # BLOCK 7: INPUT DATA AND STEERING INTERFACE
    # ==========================================

    def set_axle_data(self, P_aP0, P_bP0, r_aP0, r_bP0, a_i, tire_radius, tire_width):
        """
        Geometry and hardware of the axle.

        P_aP0 / P_bP0 : hardpoints in design position (chassis / wheel carrier)
        r_aP0 / r_bP0 : pivot of the inner knuckle and wheel centre
        a_i           : selection vector, marks the steered links
        tire_radius / tire_width : tyre radius and tyre width (plan view)
        """
        self.P_aP0 = np.asarray(P_aP0, dtype=float)
        self.P_bP0 = np.asarray(P_bP0, dtype=float)
        self.r_aP0 = np.asarray(r_aP0, dtype=float)
        self.r_bP0 = np.asarray(r_bP0, dtype=float)
        self.a_i = np.asarray(a_i, dtype=float)
        self.tire_radius = tire_radius
        self.tire_width = tire_width

        self.v_a = np.zeros_like(self.P_aP0)

    def set_simulation_data(self, steer_min, steer_max, steer_step, steer_init, steerDot):
        """
        Range, step width and prescribed velocity of the steering input.

        steer_min / steer_max / steer_step : steering range and step width
        steer_init                         : steering angle the window starts with
        steerDot                           : prescribed steering velocity omega_a
        """
        self.steer_min = steer_min
        self.steer_max = steer_max
        self.steer_step = steer_step
        self.steer_init = steer_init
        self.steerDot = steerDot

    def apply_steering(self, steer):
        """
        Rotate the chassis hardpoints marked by a_i by the steering angle phi_a
        about the pivot of the inner knuckle and hand the geometry over to the
        kinematics object.
        """
        phi_a = self.a_i * steer
        self.kine.set_geometry(self.P_aP0, self.P_bP0, self.r_aP0, self.r_bP0, phi_a)

    def steering_velocities(self):
        """
        Velocities of the chassis hardpoints that belong to the prescribed steering
        velocity. The hardpoints only rotate, so the steering velocity appears in
        omega_a, while v_a stays zero.
        """
        omega_a = self.a_i * self.steerDot
        return omega_a, self.v_a

    # ==========================================
    # BLOCK 8: START VALUE MATRIX
    # ==========================================

    @staticmethod
    def nearest_index(array, value):
        """Index of the entry of array that is closest to value."""
        return int(np.abs(np.asarray(array) - value).argmin())

    def calc_start_values(self):
        """
        Solve the position level on the coarse grid over the steering range. The
        resulting matrix is used as start vector for the solver, so that every
        slider position starts from a converged neighbour solution.
        """
        start_time = time.time()

        # sweep from the straight ahead position to both steering ends, every step
        # uses the converged solution of the previous one as start vector
        steer_plus = np.arange(0.0, self.steer_max + self.steer_step, self.steer_step)
        steer_minus = np.arange(-self.steer_step, self.steer_min - self.steer_step,
                                -self.steer_step)

        # start vector of the very first sweep, taken from the design position
        self.beta_start = np.concatenate(([0.0], self.r_bP0))
        beta_start_steer0 = self.beta_start

        beta_plus = [self._solve_position(steer) for steer in steer_plus]

        # restart the negative direction from the solution at the straight ahead position
        self.beta_start = beta_start_steer0
        beta_minus = [self._solve_position(steer) for steer in steer_minus]

        # complete matrix over the steering range, flipped so that it runs from
        # steer_max to steer_min like the curves of the spatial model
        self.steer_array = np.concatenate((steer_plus[::-1], steer_minus))
        self.beta_matrix = np.array(beta_plus[::-1] + beta_minus)

        # console output
        print("\n" + "=" * 90)
        self.print_duration("Dauer Berechnung Beta Matrix", (time.time() - start_time) * 1000)

    def _solve_position(self, steer):
        """
        Solve the position level for one steering position and keep the result as
        start vector of the next step.
        """
        self.apply_steering(steer)
        self.kine.solve_position(self.beta_start)
        self.beta_start = self.kine.beta
        return self.beta_start

    # ==========================================
    # BLOCK 9: CURRENT POSITION AND CHARACTERISTICS
    # ==========================================

    def live_movement(self, steer_curr):
        """
        Complete evaluation of the steering position of the slider, this is the one
        shown in the plan view.
        """
        start_time = time.time()

        self.steer_curr = steer_curr
        self.apply_steering(steer_curr)
        omega_a, v_a = self.steering_velocities()

        # start vector of the coarse matrix that is closest to the current position
        self.steer_index = self.nearest_index(self.steer_array, steer_curr)
        beta_start = self.beta_matrix[self.steer_index]

        self.kine.calc_geometry(beta_start, omega_a, v_a, self.steerDot)

        # console output
        self.print_status(beta_start, (time.time() - start_time) * 1000)

    # ==========================================
    # BLOCK 10: CONSOLE OUTPUT
    # ==========================================

    @staticmethod
    def print_duration(label, duration_ms):
        """One timing line of the console output."""
        print(f"{label:<31} | {duration_ms:>8.3f} ms")

    def print_status(self, beta_start, duration_ms):
        """Print position, velocities and the instant center to the console."""
        if not self.PRINT_STATUS:
            return

        def fmt_arr4(arr):
            return "[" + ", ".join([f"{x:8.4e}" for x in arr]) + "]"

        print("\n" + "=" * 90)
        print(f"{'KOMPONENTE':<17} | {'WERT'}")
        print("-" * 90)
        print(f"{'Lenkwinkel':<17} | {np.degrees(self.steer_curr):>8.3f} deg")
        print(f"{'Beta Start':<17} | {fmt_arr4(beta_start)}")
        print(f"{'Beta Val':<17} | {fmt_arr4(self.kine.beta)}")
        print(f"{'Eta Val':<17} | {fmt_arr4(self.kine.eta)}")
        if not self.kine.pure_translation:
            print(f"{'Momentanpol':<17} | {fmt_arr4(self.kine.r_M)}")
            print(f"{'Lenkuebersetzung':<17} | {self.kine.i_steer:>8.4f}")
        else:
            print(f"{'Momentanpol':<17} | reine Translation, Radius unendlich")
        print("-" * 90)
        self.print_duration("Dauer Berechnung", duration_ms)

    # ==========================================
    # BLOCK 11: SETUP OF THE FIGURE
    # ==========================================

    def setup_figure(self):
        """
        Create the plan view window and the fixed axis limits. The limits are
        derived from the design position, so that the model does not jump around
        while the slider is moved.
        """
        start_time = time.time()

        self.fig, self.ax = plt.subplots(num="2D Suspension Kinematics", figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.25)  # space for the slider

        self._calc_axis_limits()

        self.print_duration("Dauer Plot Setup", (time.time() - start_time) * 1000)
        print("=" * 90 + "\n")

    def _calc_axis_limits(self):
        """Fixed limits of the plan view, large enough for all hardpoints and the tyre."""
        points = np.vstack([self.P_aP0, self.P_bP0, self.r_aP0, self.r_bP0])

        # the tread rectangle rotates with the wheel carrier, its envelope is
        # therefore a circle of half its diagonal around the wheel centre
        wheel_envelope = 0.5 * np.hypot(2.0 * self.tire_radius, self.tire_width)

        x_min = min(points[:, 0].min(), self.r_bP0[0] - wheel_envelope) - self.PLOT_MARGIN
        x_max = max(points[:, 0].max(), self.r_bP0[0] + wheel_envelope) + self.PLOT_MARGIN
        y_min = min(points[:, 1].min(), self.r_bP0[1] - wheel_envelope) - self.PLOT_MARGIN * 2
        y_max = max(points[:, 1].max(), self.r_bP0[1] + wheel_envelope)

        self.xlim = (x_min, x_max)
        self.ylim = (y_min, y_max)

    # ==========================================
    # BLOCK 12: UPDATE OF THE FIGURE AND INTERACTIVE CONTROLS
    # ==========================================

    def update_figure(self):
        """Redraw the plan view of the current steering position."""
        start_time = time.time()

        self._reset_axes()
        self._draw_wheel()
        self._draw_links()
        self._draw_wheel_carrier()
        self._draw_inner_knuckle()
        self._draw_chassis_joints()
        self._draw_instant_center()
        self._set_title()

        self.fig.canvas.draw_idle()

        self.print_duration("Dauer Plot Update", (time.time() - start_time) * 1000)
        print("=" * 90 + "\n")

    def _reset_axes(self):
        """Clear the axes and restore the fixed limits and labels."""
        self.ax.cla()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_xlabel('x [mm]', fontsize=12)
        self.ax.set_ylabel('y [mm]', fontsize=12)

    def _draw_links(self):
        """The three links as straight lines between chassis and wheel carrier."""
        for P_a, P_b in zip(self.kine.P_a, self.kine.P_b):
            self.ax.plot([P_a[0], P_b[0]], [P_a[1], P_b[1]], color=self.COLOR_LINK,
                         linestyle='-', linewidth=self.LINE_WIDTH, zorder=3)

    def _draw_chassis_joints(self):
        """
        Joints on the chassis side: the steered hardpoints, the fixed hardpoint of
        the trailing link and the pivot of the inner knuckle.
        """
        steered = self.a_i != 0.0
        P_a = self.kine.P_a
        self.ax.scatter(P_a[steered, 0], P_a[steered, 1], color=self.COLOR_CHASSIS,
                        s=self.MARKER_SIZE, marker='o', zorder=5)
        self.ax.scatter(P_a[~steered, 0], P_a[~steered, 1], color=self.COLOR_CHASSIS_FIXED,
                        s=self.MARKER_SIZE, marker='o', zorder=5)
        self.ax.scatter(self.r_aP0[0, 0], self.r_aP0[0, 1], color=self.COLOR_PIVOT,
                        s=self.MARKER_SIZE, marker='o', zorder=5)

    def _draw_wheel_carrier(self):
        """
        Body of the wheel carrier as a polyline through its hardpoints. The
        hardpoint of the trailing link lies between the two of the transverse
        links, so the polyline runs P_b,1 - P_b,0 - P_b,2.
        """
        P_b = self.kine.P_b[[1, 0, 2]]
        self.ax.plot(P_b[:, 0], P_b[:, 1], color=self.COLOR_WHEEL_CARRIER,
                     linestyle='-', linewidth=self.LINE_WIDTH, zorder=4)
        self.ax.scatter(self.kine.P_b[:, 0], self.kine.P_b[:, 1],
                        color=self.COLOR_WHEEL_CARRIER, s=self.MARKER_SIZE,
                        marker='o', zorder=5)

    def _draw_inner_knuckle(self):
        """Body of the inner knuckle: pivot connected to every steered hardpoint."""
        r_a = self.r_aP0[0]
        for index in np.flatnonzero(self.a_i):
            P_a = self.kine.P_a[index]
            self.ax.plot([r_a[0], P_a[0]], [r_a[1], P_a[1]], color=self.COLOR_CHASSIS,
                         linestyle='-', linewidth=self.LINE_WIDTH, zorder=4)

    def _draw_instant_center(self):
        """Instant center of the wheel carrier, undefined for a pure translation."""
        if self.kine.pure_translation:
            return
        self.ax.scatter(self.kine.r_M[0], self.kine.r_M[1],
                        color=self.COLOR_INSTANT_CENTER, s=self.MARKER_SIZE,
                        marker='o', zorder=5)

    def _draw_wheel(self):
        """
        Plan view of the tyre: a rectangle of tyre diameter and tyre width, rotated
        with the wheel carrier and centred on the wheel centre r_b.
        """
        length, width = 2.0 * self.tire_radius, self.tire_width
        b_box = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
            [-length / 2, -width / 2],
            ])

        # the corners are body fixed vectors of K_b, so they are rotated by {}^{0b}T
        box = (self.kine.T_0b @ b_box.T).T + self.kine.r_b

        self.ax.plot(box[:, 0], box[:, 1], color=self.COLOR_TIRE, alpha=0.5,
                     linewidth=3, zorder=2)
        self.ax.fill(box[:, 0], box[:, 1], color=self.COLOR_TIRE, alpha=0.2, zorder=1)

    def _set_title(self):
        """Steering angle of the inner knuckle and resulting angle of the outer knuckle."""
        self.ax.set_title(f"inner knuckle {np.degrees(self.steer_curr):.2f} degrees   |   "
                          f"outer knuckle {np.degrees(self.kine.beta[0]):.2f} degrees",
                          fontsize=14, fontweight='bold')

    def add_controls(self):
        """
        Turn the figure into an interactive plot: a slider for the steering input
        and a shortcut to close the window.
        """
        ax_slider = self.fig.add_axes([0.25, 0.1, 0.5, 0.03])
        self.steer_slider = Slider(ax_slider, 'inner knuckle [rad]',
                                   self.steer_min, self.steer_max,
                                   valinit=self.steer_init,
                                   valstep=self.steer_step / self.FINE_FACTOR,
                                   valfmt='%1.4f')

        def recalculate_and_redraw(val):
            self.live_movement(val)
            self.update_figure()

        self.steer_slider.on_changed(recalculate_and_redraw)
        self._connect_events()

        # initial call at start up
        recalculate_and_redraw(self.steer_init)

    def _connect_events(self):
        """Close the window on 'q' or 'escape'."""
        def on_key_press(event):
            if event.key in ['q', 'escape']:
                plt.close(self.fig)

        self.fig.canvas.mpl_connect('key_press_event', on_key_press)


# ==========================================
# BLOCK 13: MAIN PROGRAM
# ==========================================

def build_simulation(kine_object):
    """Interactive planar simulation of the five-link axle with a rotating inner knuckle."""
    simulation = LiveMovement2d(kine_object)
    simulation.set_axle_data(P_aP0, P_bP0, r_aP0, r_bP0, a_i, tire_radius, tire_width)
    simulation.set_simulation_data(steer_min, steer_max, steer_step, steer_init, steerDot)
    return simulation


# The simulation is built at import time, so that it is available in an
# interactive console as well.
kine = Kinematics2d()
simulation = build_simulation(kine)


if __name__ == "__main__":

    simulation.calc_start_values()   # coarse start value matrix over the steering range
    simulation.setup_figure()        # create the plan view window
    simulation.add_controls()        # slider and the first update

    plt.show(block=True)