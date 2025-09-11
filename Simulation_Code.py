import rebound
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.ticker as ticker
import os


G = 4*np.pi**2
M_bh = 10.0         
M_planet = 1/332946 
R_planet = 0.000002
a = 0.02            

R_in = 2e-7
R_out = 0.02072
H_acc = 0.7 * R_out
acc_m = 0.15
r0 = 0.1


# and the ecc the direction

def simulate(ecc, direction):

    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.G = G
    sim.integrator = "ias15"


    sim.add(m=M_bh)                     
    r_peri = 0.02
    a_eff = r_peri / (1 - ecc)
    sim.add(m=M_planet, a=a_eff, e=ecc)

    sim.move_to_com()


    orbital_period = 2*np.pi*np.sqrt(a_eff**3 / (G * (M_bh+M_planet)))
    print(f"Orbital period = {orbital_period:.3e} years")


    roche_limit = R_planet * ((M_bh/M_planet) ** (1/3))
    print(f"Roche limit = {roche_limit} AU")

    # SIMULATION TIMESPAN
    # sim.dt = orbital_period / 1000
    # print(f"Chosen timestep = {sim.dt:.3e} years")

    N_orbits = 10000
    tmax = N_orbits * orbital_period


    def samples_per_orbit_for_ecc(e):
        return int(np.interp(e, [0.0, 0.3, 0.6, 0.85, 0.95],
                            [200,  400,  800, 2000, 3000]))

    samples = samples_per_orbit_for_ecc(ecc)
    N_outputs = samples * N_orbits
    times = np.linspace(0, tmax, N_outputs)



    planet = sim.particles[1]
    bh = sim.particles[0]


    def disk_gravity(reb_sim):
        sim = reb_sim.contents
        bh = sim.particles[0]
        planet = sim.particles[1]

        r = np.sqrt((planet.x - bh.x)**2 + (planet.y - bh.y)**2)
        kappa = 0.2

        if r <= R_in:
            M_in = 0.0
            M_out = acc_m
        elif r >= R_out:
            M_in = acc_m
            M_out = 0.0
        else: 
            I_total = (R_out**0.5 - R_in**0.5) / 0.5
            Sigma0 = acc_m / (2*np.pi * (r0**1.5) * I_total)

            I_in = (r**0.5 - R_in**0.5) / 0.5
            M_in = 2*np.pi * Sigma0 * (r0**1.5) * I_in

            M_out = acc_m - M_in

        a_radial = -G * (M_in - kappa * M_out) / (r**2)

        dx = planet.x - bh.x
        dy = planet.y - bh.y
        r = np.hypot(dx, dy)

        ux, uy = dx/r, dy/r
        planet.ax += a_radial * ux
        planet.ay += a_radial * uy

    def gas_drag(reb_sim):
        sim = reb_sim.contents
        bh = sim.particles[0]
        planet = sim.particles[1]

        dx = planet.x - bh.x
        dy = planet.y - bh.y
        r  = np.hypot(dx, dy)

        if R_in <= r <= R_out:

            p0   = 181
            p_acc = p0 * (r/r0) ** (-1.5)

            tx = -dy / r
            ty =  dx / r

            v_kep = direction * np.sqrt(G*(M_bh+M_planet)/r)

            h    = 0.7
            p    = 1.5
            q    = 0.75
            eta  = 0.5*(p+q)*h*h
            v_gas_mag = (1.0 - eta) * v_kep

            vg_x, vg_y = v_gas_mag*tx, v_gas_mag*ty

            dvx = planet.vx - vg_x
            dvy = planet.vy - vg_y
            v_rel2 = dvx*dvx + dvy*dvy
            v_rel  = np.sqrt(v_rel2 + 1e-16)

            Cd   = 1.0
            Area = np.pi * R_planet**2
            m    = M_planet
            Force = 0.5 * Cd * p_acc * Area * v_rel2

            planet.ax += -Force/m * (dvx / v_rel)
            planet.ay += -Force/m * (dvy / v_rel)


    def combined_forces(reb_sim):
        disk_gravity(reb_sim)
        gas_drag(reb_sim)


    sim.additional_forces = combined_forces





    # -------------------
    # Storage arrays
    # -------------------
    x, y = [], []
    eccs = []
    perih_times = []
    perih_distances = []
    years = []
    days = []
    distance = []



    last_r = None
    decreasing = True  

    for t in tqdm(times):
        r = np.sqrt((planet.x - bh.x)**2 + (planet.y - bh.y)**2)

        if r - 0.0001 <= roche_limit:
            print(f"Planet has been disintegrated at {t} years")
            x.append(planet.x)
            y.append(planet.y)
            eccs.append(planet.e)
            perih_times.append(t)
            perih_distances.append(r)
            years.append(t)
            days.append(t * 365.25)
            distance.append(np.sqrt((planet.x - bh.x)**2 + (planet.y - bh.y)**2))

            break
        else:
            sim.integrate(t)

            x.append(planet.x)
            y.append(planet.y)
            eccs.append(planet.e)
            years.append(t)
            days.append(t * 365.25)
            distance.append(np.sqrt((planet.x - bh.x)**2 + (planet.y - bh.y)**2))

            if last_r is not None:
                if decreasing and r > last_r:
                    perih_times.append(t)
                    perih_distances.append(last_r)
                    decreasing = False
                elif r < last_r:
                    decreasing = True

            last_r = r


    print("Simulation finished :))))))")



    out_dir = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Results"
    os.makedirs(out_dir, exist_ok=True)  
    
    df_xy = pd.DataFrame({"x": x, "y": y})
    df_xy.to_csv(os.path.join(out_dir, f"xy_({ecc}_{direction}).csv"), index=False)
    print("XYs saved")
    
    df_ecc = pd.DataFrame({"years": years, "eccentricity": eccs})
    df_ecc.to_csv(os.path.join(out_dir, f"eccentricities_({ecc}_{direction}).csv"), index=False)
    print("Eccentricities saved")

    df_dist = pd.DataFrame({"years": (perih_times), "distance [AU]": perih_distances})
    df_dist.to_csv(os.path.join(out_dir, f"planet_distances_({ecc}_{direction}).csv"), index=False)
    print("Planet-BH distances saved")

    df_constdist = pd.DataFrame({"years": years, "distance": distance})
    df_constdist.to_csv(os.path.join(out_dir, f"constant_distances_({ecc}_{direction}).csv"), index=False)
    print("Planet-BH distances saved")


ecccs = np.linspace(0.0,0.9,19)

directionss = [-1, 1]
print(ecccs)
for i in ecccs:
    for j in directionss:
        simulate(i,j)
