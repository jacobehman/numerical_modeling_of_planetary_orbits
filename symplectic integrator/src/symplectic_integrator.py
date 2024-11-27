'''
    Project 1 - Symplectic Integrator 
    EAPS 55501 - Jacob Ehman

    Utilizes Democratic Heliocentric Method given by Duncan, Levison, and Lee (1998).

    --------Reference--------
    Duncan, M.J., Levison, H.F., Lee, M.H., 1998. A Multiple Time Step Symplectic Algorithm for Integrating Close
        Encounters. AJ 116, 2067-2077. https://doi.org/10.1086/300541
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import pandas as pd
from tqdm import tqdm

# Global constants (SI)
AU2M = const.au.value
GMSunSI = const.GM_sun.value
MSun = const.M_sun.value
JULIAN_DAY = 365.25
JD2S = 86400
YR2S = np.double(JULIAN_DAY * JD2S)
GC_AU = const.G.value * (YR2S ** 2)/(AU2M ** 3) * MSun

class Symplectic:

    def __init__(self, bodies):

        self.t = 0.0
        self.tindex = 0.0

        cols = ['t','rH', 'vH', 'E', 'M', 'e', 'long_peri']
        self.bodies = []
        self.masses = []

        for body in bodies:

            if body['name'] == 'Sun': 
                self.sunmass = body['mass']

            self.bodies.append(body['name'])
            self.masses.append(body['mass'])

        self.n = len(self.bodies) - 1

        self.masses = np.vstack(self.masses)

        multi_index_cols = pd.MultiIndex.from_product([self.bodies, cols], names=['Planet', 'Columns'])
        self.data = pd.DataFrame(columns = multi_index_cols)
        
        rhlist = []
        vhlist = []
        # load in initial positions and velocities
        for body in bodies:

            rhlist.append(np.array([body['r0'][0], body['r0'][1], body['r0'][2]]))
            vhlist.append(np.array([body['v0'][0], body['v0'][1], body['v0'][2]]))

        # Intitialize working vectors with initial values
        self.rhvec = np.vstack(rhlist)
        self.vhvec = np.vstack(vhlist)

        self.masses = np.array(self.masses)
        
        self.vHvBconv(direction='toBary')
        return
    
    def integrate(self, final_t, dt, save_cadence):

        self.dt = dt

        rhhist = [np.copy(self.rhvec[1:])]
        vhhist = [np.copy(self.vhvec[1:])]
        thist = [self.t]

        for i in tqdm(range(int(np.ceil(final_t/dt/save_cadence)))):

            for j in range(save_cadence):
                self.tindex = i
            
                # Step
                self.step()
                self.t += self.dt

            self.vHvBconv(direction='toHelio')

            rhhist.append(np.copy(self.rhvec[1:]))
            vhhist.append(np.copy(self.vhvec[1:]))
            thist.append(self.t)


        self.collectdata(thist, rhhist, vhhist)
        return

    def step(self):

        self.drift(0.5*self.dt)
        self.kick(0.5*self.dt)
        self.kepler(self.dt)
        self.kick(0.5*self.dt)
        self.drift(0.5*self.dt)

        return

    def drift(self, dt):

        pt = np.sum(self.masses * self.vbvec, axis=0) / self.sunmass
        self.rhvec[1:] += pt*dt

        return

    def kick(self, dt):
        vbdot = []

        # Calculate interaction accelerations
        for body in self.bodies:
            vbdot.append(self.solve_vbdot(body))
        
        self.vbdot = np.vstack(vbdot)

        # Kick barycentric velocity
        self.vbvec += self.vbdot * dt

        return
    
    def kepler(self, dt):

        # Not kepler drifting the sun
        vbvec = [self.vbvec[0]]
        rhvec = [self.rhvec[0]]

        for body in self.bodies:
            planet_idx = self.bodies.index(body)

            if body != 'Sun':
                # current (initial) position & barycentric velocity vectors
                rhvec0 = self.rhvec[planet_idx]
                vbvec0 = self.vbvec[planet_idx]

                mubody = GC_AU*(self.masses[planet_idx]+self.sunmass)

                # Calculate drift of body
                rhvecnew, vbvecnew = Symplectic.scalar_drift(np.array(rhvec0),np.array(vbvec0), mubody, dt)
                vbvec.append(vbvecnew)
                rhvec.append(rhvecnew)

        # Save rvec and vbvec kepler drifts
        self.vbvec = np.vstack(vbvec)
        self.rhvec = np.vstack(rhvec)

        return 

    def solve_vbdot(self, body):

        vbdot = 0

        # 'planet' is what given body is interacting with (body j vs body i)
        for planet in self.bodies[1:]:
            if body != planet:
                bodyi_idx = self.bodies.index(body)
                bodyj_idx = self.bodies.index(planet)

                drbody = self.rhvec[bodyj_idx] - self.rhvec[bodyi_idx]
                vbdot += self.masses[bodyj_idx] *  drbody / np.linalg.norm(drbody)**3

        return GC_AU*vbdot

    def vHvBconv(self, direction):
        mtot = np.sum(self.masses)

        # output based on input direction 
        if direction == 'toBary':
            vbarycenter = -np.sum(self.masses * self.vhvec, axis=0) / mtot
            self.vbvec = self.vhvec + vbarycenter
        elif direction == 'toHelio':
            vbarycenter = -np.sum(self.masses * self.vbvec, axis=0) / self.sunmass
            self.vhvec = self.vbvec - vbarycenter
        else:
            raise RuntimeError('helio-barycentric velocity input not supported.')
        return
    
    @staticmethod 
    def scalar_drift(rvec0, vvec0, mu, dt):

        # Calculate some orbital elements
        rmag0 = np.linalg.norm(rvec0)
        vmag2 = np.vdot(vvec0,vvec0)
        h = np.cross(rvec0,vvec0)
        hmag2 = np.vdot(h,h)
        a = (2.0 / rmag0 - vmag2/mu)**(-1)
        e = np.sqrt(1 - hmag2 / (mu * a))

        n = np.sqrt(mu / a**3)  # Kepler's 3rd law to get the mean motion

        # Current eccentric anomaly
        E0 = np.where(e > np.finfo(np.float64).tiny, np.arccos(-(rmag0 - a) / (a * e)), 0)

        if e>1:
            raise RuntimeError('Planet escaped - hyperbolic trajectory')
        
        # Correct for periapsis/apoapsis directions
        if np.sign(np.vdot(rvec0, vvec0)) < 0.0:
            E0 = 2 * np.pi - E0

        M0 = E0 - e * np.sin(E0)
        M = M0 + n * dt

        E = Symplectic.danby(M, e, tol=1e-12)
        dE = E - E0

        f = a / rmag0 * (np.cos(dE) - 1.0) + 1.0
        g = dt + (np.sin(dE) - dE) / n

        rvec = f* rvec0 + g * vvec0
        rmag = np.linalg.norm(rvec)

        fdot = -a**2 / (rmag * rmag0) * n * np.sin(dE)
        gdot = a / rmag * (np.cos(dE) - 1.0) + 1.0

        vvec = fdot * rvec0 + gdot * vvec0
        return rvec, vvec

    @staticmethod
    def danby(M, e, tol=1e-12):
        
        # Initial guess from Danby & Burkardt (1983)
        k = 0.85
        E = [M + np.sign(np.sin(M)) * k * e]

        def f(E):
            return E - e*np.sin(E) - M
        def fprime(E):
            return 1 - e*np.cos(E)
        def f2prime(E):
            return e*np.sin(E)
        def f3prime(E):
            return e*np.cos(E)
        
        
        i = 0
        diff = 1
        while abs(diff) >= tol:
            
            del1 = - f(E[i])/fprime(E[i])
            del2 = - f(E[i])/(fprime(E[i]) + 0.5*del1*f2prime(E[i]))
            del3 = - f(E[i])/(fprime(E[i]) + 0.5*del2*f2prime(E[i]) + (1/6)* del2**2 * f3prime(E[i]))

            E_next = E[i] + del3 
            E.append(E_next)

            diff = (E_next - E[i]) / E_next
            i += 1
            MAXLOOPS = 50
            if i > MAXLOOPS: 
                raise RuntimeError("The danby function did not converge on a solution.")

        return E[-1]

    @staticmethod
    def orbElms(rvec,vvec,mu):
        
        R = np.linalg.norm(rvec)
        V2 = np.vdot(vvec,vvec)

        h = np.cross(rvec,vvec)
        H = np.linalg.norm(h)

        rdot = np.sign(np.vdot(rvec,vvec)) * np.sqrt(V2 - (H/R)**2)

        a = (2/R - V2/mu)**(-1)
        e = np.sqrt(1 - H**2/(mu*a))
        I = np.arccos(h[2]/H)
        
        sinLgAsc =  np.sign(h[2]) * h[0]/(H*np.sin(I))
        cosLgAsc = -np.sign(h[2]) * h[1]/(H*np.sin(I))

        sinf = a*(1-e**2)/(H*e) * rdot
        cosf = 1/e * (a*(1-e**2)/R - 1)

        # Argument of pericentere + true anomaly
        sinwf = rvec[2]/(R*np.sin(I))
        coswf = (1/cosLgAsc)*(rvec[0]/R + sinLgAsc*sinwf*np.cos(I))

        # Final values
        LgAsc = np.arctan2(sinLgAsc,cosLgAsc)
        f = np.arctan2(sinf,cosf)
        wf = np.arctan2(sinwf,coswf)
        w = wf - f
        long_peri = w + LgAsc

        E = np.where(e > np.finfo(np.float64).tiny, np.arccos(-(R - a) / (a * e)), 0)
        if np.sign(np.vdot(rvec, vvec)) < 0.0:
            E = 2 * np.pi - E

        M = E - e * np.sin(E)

        energy = V2/2 - mu/R

        return e, np.rad2deg(M), np.rad2deg(long_peri), energy
    
    def collectdata(self, thist, rhhist, vhhist):

        for planet in self.bodies[1:]:
            ehist = []
            Mhist = []
            varpihist = []
            Ehist = []

            planet_idx = self.bodies.index(planet) - 1
            muplanet = GC_AU*(self.masses[planet_idx]+self.sunmass)

            self.data[(planet, 'rH')] = [rhvec[planet_idx] for rhvec in rhhist]
            self.data[(planet, 'vH')] = [vhvec[planet_idx] for vhvec in vhhist]
            self.data[(planet, 't')] = thist

            for i in range(len(thist)):
                rvec = self.data.loc[i, (planet, 'rH')]
                vvec = self.data.loc[i, (planet, 'vH')]
                e, M, varpi, E = Symplectic.orbElms(rvec, vvec, muplanet)
                ehist.append(e)
                Mhist.append(M)
                varpihist.append(varpi)
                Ehist.append(E)

            self.data[(planet, 'e')] = ehist
            self.data[(planet, 'M')] = Mhist
            self.data[(planet, 'varpi')] = varpihist
            self.data[(planet, 'E')] = Ehist

        self.thist = thist
        return

    def plot(self):
        lambda_N = self.data[('Neptune', 'M')] + self.data[('Neptune', 'varpi')]
        lambda_P = self.data[('Pluto', 'M')] + self.data[('Pluto', 'varpi')]
        res_angle = np.mod(3*lambda_P - 2*lambda_N - self.data[('Pluto', 'varpi')],360)

        total_energy = np.nansum(self.data.loc[:, (slice(None), 'E')].values, axis=1)
        total_energy = (total_energy - total_energy[0]) / total_energy[0]

        plt.figure(1)
        plt.plot(self.thist, res_angle)
        plt.xlabel('Time [yr]')
        plt.ylabel('Resonance Angle [deg]')
        plt.title('Neptune-Pluto 3:2 Resonance')
        plt.tight_layout()

        thisdir = os.getcwd()
        parentdir = os.path.abspath(os.path.join(thisdir,os.pardir))
        plotdir = os.path.join(parentdir,'plots')

        save_path = os.path.join(plotdir, f'res_angle')
        plt.savefig(save_path,dpi=300)

        plt.figure(2)
        plt.plot(self.thist, total_energy)
        plt.xlabel('Time [yr]')
        plt.ylabel(r'$\frac{\Delta\epsilon}{\epsilon_{0}}$')
        plt.title('Change in Specific Energy over Time')
        plt.tight_layout()

        save_path = os.path.join(plotdir, f'energy')
        plt.savefig(save_path,dpi=300)
        print('Plots saved.')
        return

# AU/DAY R0/V0 UNITS
solar_system_bodies = [
    {"name": "Sun", "mass_kg": MSun, "r0": (0, 0, 0), "v0": (0, 0, 0)},
    #{"name": "Mercury", "mass_kg": 3.3011e23, "r0": (3.347e-01, -2.106e-01, -4.792e-02), "v0": (9.457e-03, 2.510e-02, 1.183e-03)},  
    #{"name": "Venus", "mass_kg": 4.8675e24, "r0": (-4.641e-01, 5.473e-01, 3.428e-02), "v0": (-1.549e-02, -1.319e-02, 7.136e-04)},
    #{"name": "Earth", "mass_kg": 5.97237e24, "r0": (7.844e-01, 6.083e-01, -1.996e-05), "v0": (-1.081e-02, 1.352e-02, 2.330e-07)},
    #{"name": "Mars", "mass_kg": 6.4171e23, "r0": (3.248e-01, -1.392e+00, -3.714e-02), "v0": (1.415e-02, 4.3809e-03, -2.558e-04)},
    #{"name": "Jupiter", "mass_kg": 1.8982e27, "r0": (1.873e+00, 4.683e+00, -6.137e-02), "v0": (-7.104e-03, 3.164e-03, 1.458e-04)},
    #{"name": "Saturn", "mass_kg": 5.6834e26, "r0": (-8.251e+00, -5.225e+00, 4.193e-01), "v0": (2.677e-03, -4.723e-03, -2.458e-05)},
    #{"name": "Uranus", "mass_kg": 8.6810e25, "r0": (1.992e+01, 2.342e+00, -2.493e-01), "v0": (-4.936e-04, 3.724e-03, 2.027e-05)},
    {"name": "Neptune", "mass_kg": 1.02413e26, "r0": (2.647e+01, -1.409e+01, -3.196e-01), "v0": (1.449e-03, 2.79e-03, -9.063e-05)},
    {"name": "Pluto", "mass_kg": 1.303e22, "r0": (4.913e+00, -3.188e+01, 1.991e+00), "v0": (3.156e-03, -1.458e-04, -8.987e-04)}
]

# Convert from AU/day to AU/year
for body in solar_system_bodies:
    body['v0'] = tuple(vel * JULIAN_DAY for vel in body['v0'])
    body['mass'] = body['mass_kg']/MSun

sim = Symplectic(solar_system_bodies)
sim.integrate(final_t=1e5, dt=5, save_cadence=10)
sim.plot()