from dataclasses import dataclass

import numpy as np
import xarray as xr
import gsw
from scipy import minimize


@dataclass
class FlightParameters:
    """
    Configuration parameters that are assumed constant
    over a deployment or dive.
    """
    # flight model coefficients
    hd_a: float = 0.004  # 
    hd_b: float = 0.01
    hd_c: float = 5.7e-5
    xl: float = 1.8
    mass: float = 52.0 # kg
    volmax: float = 51000 # cc

    # reference density
    rho0: float = 1027.5 # kg/m^3
    gravity = 9.81 # m/s^2

    # buoyancy model
    c_vbd: float = 0.0
    vbd_bias: float = 0.0
    vbd_min_cnts: float = 0.0
    vbd_cnts_per_cc: float = -4.077

    # glider volume model
    temp_ref: float = 15.0
    therm_exp: float = 0.0           # thermal expansion
    abs_compress: float = 0.0            # compressibility

    # location
    lon: float = 0.0
    lat: float = 0.0

    # solver
    tol: float = 1e-3
    max_iter: int = 15


class SteadyFlightModel:
    """
    Steady-state glider flight model.

    Paramters:
    ----------
    params: FlightParameters
        Configuration parameters for the flight model.
    Input:
    ------
    ds: xarray.Dataset
        Input dataset. Must contain the following variables:
        - PSAL: Practical Salinity (unitless)
        - TEMP: In-situ Temperature (°C)
        - PRESS: Pressure (dbar)
        - vbd: Variable Buoyancy Device volume (cc)
        - pitch: Glider pitch angle (degrees)

    Returns:
    -------
    ds: xarray.Dataset
        Input dataset with added variables for flight speed and angle.
    """

    def __init__(self, params: FlightParameters) -> None:
        self.params = params


    def solve_model(self, ds: xr.Dataset, x0: np.ndarray) -> xr.Dataset:
        """
        Complete workflow.
        """
        print("🚁 Starting Glider Flight Model Parameter Optimization")
        print("=" * 60)

        result = minimize(
            self.func_to_minimize(ds),
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': self.params.max_iter, 'xatol': self.params.tol, 'fatol': self.params.tol, 'disp': True}
        )
        
    
    def func_to_minimize(self, ds):

        updn = xr.where(ds.PROFILE_NUMBER % 2 == 1, 1, -1) # 1 for up, -1 for down
        dzdt = ds.GLIDER_VERT_VELO_DZDT.values

        rho = self.compute_density(ds)
        vol = self.compute_volume(ds)

        F_B = self.compute_buoyancy_force(
            density=rho,
            vol=vol,
        )

        umag, thdeg = self.solve_flight(
            buoy=F_B,
            pitch=ds.pitch.values,
        )

        return ds.assign(
            umag=("N_MEASUREMENTS", umag),
            thdeg=("N_MEASUREMENTS", thdeg),
        )
    
    def compute_density(self, ds):
        """Compute in-situ density from salinity, temperature, and pressure using GSW.

        Parameters
        ----------
        ds : xarray.Dataset

        Returns
        -------
        density : array-like
            In-situ density (kg/m³).
        """
        salin = ds.PSAL.values
        temp = ds.TEMP.values
        press = ds.PRES.values

        # Step 1: Convert Practical Salinity to Absolute Salinity
        SA = gsw.SA_from_SP(salin, press, self.params.lon, self.params.lat)

        # Step 2: Convert in-situ Temperature to Conservative Temperature
        CT = gsw.CT_from_t(SA, temp, press)

        # Step 3: Compute in-situ density
        density = gsw.rho(SA, CT, press)

        return density
    
    
    def compute_volume(self, ds):
        """Compute glider volume from VBD and other parameters.

        Parameters
        ----------
        ds : xarray.Dataset

        Returns
        -------
        volume : array-like
            Glider volume (cc).
        """
        vbd = ds.VBD.values
        press = ds.PRES.values
        temp = ds.TEMP.values
        p = self.params
        
        vol1 = vbd + p.volmax + (p.c_vbd - p.vbd_min_cnts) / p.vbd_cnts_per_cc
        compr_factor = np.exp(-p.abs_compress * press + p.therm_exp * (temp - p.temp_ref))

        #vbdc = vbd - p.vbd_bias
        vol = (vol1 - p.vbd_bias) * compr_factor
        return vol
    

    def compute_buoyancy_force(
        self,
        density: np.ndarray,
        vol: np.ndarray,
        ) -> np.ndarray:
        """
        Compute buoyancy force B from density, volume, and other parameters.
        """

        p = self.params
        cc_to_m3 = 1e-6 # conversion factor from cc to m³

        F_B = p.gravity * (- p.mass + density * vol * cc_to_m3)

        return F_B
    
    def compute_lift_force(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Compute lift force L from density, speed, and other parameters.
        """

        p = self.params

        F_L = q * p.xl**2 * p.hd_a * np.sin(alpha)

        return F_L
    
    def compute_drag_force(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """
        Compute drag force D from density, speed, and other parameters.
        """

        p = self.params

        F_D = q * p.xl ** 2 * (p.hd_b * q ** -0.25 + p.hd_c * alpha ** 2)

        return F_D
    

    def compute_alpha(
        self,
        q: np.ndarray,
        alpha: np.ndarray,
        pitch: np.ndarray,
        updn: np.ndarray,
    ) -> np.ndarray:
        """
        Compute alpha from density, speed, and other parameters.
        """

        p = self.params

        alpha = updn * (p.hd_b * q ** -0.25 + p.hd_c * alpha ** 2) / (p.hd_a * np.tan(np.deg2rad(pitch - alpha)))

        return alpha

    def solve_flight(
        self,
        F_B: np.ndarray,
        pitch: np.ndarray,
        updn: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Iterative q-solver.
        """

        p = self.params

        th = (np.pi / 4) * updn
        # Initial guess for dynamic pressure q based on free fall assumption
        q = (F_B / (p.xl**2 * p.hd_b)) ** (4 / 3)

        q_old = np.zeros_like(q)

        alpha = np.zeros_like(q)

        thdeg = np.zeros_like(q)

        param = np.ones_like(q)

        valid = ((F_B != 0) & (np.sign(F_B) * np.sign(pitch) > 0))

        iteration = 0

        while (np.any(np.abs((q[valid] - q_old[valid])/ q[valid]) > p.tol) and iteration < p.max_iter):

            q_old = q.copy()

            param_inv = (p.hd_a**2 * np.tan(th)**2 * q**0.25 / (4 * p.hd_b * p.hd_c))

            valid = ((param_inv > 1) & (np.sign(F_B) * np.sign(pitch) > 0))

            param[valid] = (4 * p.hd_b * p.hd_c / ( p.hd_a**2 * np.tan(th[valid])**2 * q[valid]**0.25 ))

            q[valid] = (F_B[valid] * np.sin(th[valid]) / (2 * p.xl**2 * p.hd_b * q[valid]**(-0.25))) * (1 + np.sqrt(1 - param[valid]))

            q = np.maximum(q, 1e-10)

            alpha[valid] = (
                -p.hd_a
                * np.tan(th[valid])
                / (2 * p.hd_c)
            ) * (
                1
                - np.sqrt(
                    1 - param[valid]
                )
            )

            thdeg[valid] = (
                pitch[valid]
                - alpha[valid]
            )

            stall = (
                (param_inv <= 1)
                | (
                    np.sign(F_B)
                    * np.sign(pitch)
                    < 0
                )
            )

            q[stall] = 0.0
            thdeg[stall] = 0.0

            th = np.deg2rad(thdeg)

            iteration += 1

        umag = (
            100
            * np.sqrt(
                2 * q / p.rho0
            )
        )

        return umag, thdeg