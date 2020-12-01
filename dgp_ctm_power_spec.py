## Import the required python packages

import numpy as np
from ctm import CTM

from scipy.interpolate import interp1d as interp
from scipy.integrate import quad



## Define k values and z values

k_vals = np.logspace(-3, np.log10(0.9), 1000)

z_vals = np.linspace(0, 200, 1000)



## Scale factor a(z) for DGP gravity

def scale_factor(z_val):

    return (1+z_val)**(-1)


## Linear growth factor D(z) for DGP gravity

def D1_dgp (z_vals, alpha, Ho=67.37, omega0_m=(0.11933+0.02242)/(0.6737**2), omega0_lamda=1.0-(0.11933+0.02242)/(0.6737**2)):

    def rc(alpha, Ho=67.37, omega0_m=omega0_m):         # cross-over scale rc for DGP gravity

        return ((1-omega0_m)**(1/(alpha-2)))/Ho

    def hubble_dgp(z_val, alpha, Ho=67.37, omega0_m=omega0_m, omega0_lamda=omega0_lamda):         # Hubble parameter for DGP gravity

        A = (Ho**2)*((omega0_m/(scale_factor(z_val)**3)))
        B = np.sqrt(A)**alpha/((rc(alpha))**(2-alpha))
        H = (A+B)**(1/2)

        return H

## Solving integral

    def integrand(zz, alpha):

        return (1+zz)/(hubble_dgp(zz, alpha))**3

    def I(z, alpha):

        return quad(integrand, z, 150, args=(alpha))[0]

## Solving D(z)

    def linear_growth_factor(z_val, alpha, Ho=67.37, omega0_m=(0.11933+0.02242)/(0.6737**2)):

        return ((5*omega0_m*Ho**2)/2)*hubble_dgp(z_val, alpha)*I(z_val, alpha)

    def D1_dgp(z_vals, alpha):

        D=np.zeros_like(z_vals)

        for i in range(len(z_vals)):

            D[i] = linear_growth_factor(z_vals[i], alpha=alpha)

        D_vals = interp(z_vals, D)

        D_norm = D/D_vals(0.0)         # Normalise D(z) at 1 to today

        return D_norm

    return D1_dgp(z_vals, alpha)



##  input_A, A(t), for the CTM code

D1_dgp_array=D1_dgp(z_vals, 1.0)

D1_dgp_func=interp(z_vals, D1_dgp_array)

A_vals=D1_dgp_func(z_vals)/D1_dgp_func(99.0)



## input_k and input_P, initial power spectrum input for the CTM code

input_k=np.logspace(-5, 2, 5000)

input_P=D1_dgp_func(99.0)**2*CTM().linear_power(input_k)



## Calc undamped power spec for DGP, using the CTM code

P_ctm_dgp=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k)

P_ctm_dgp_1=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=1.0)

P_ctm_dgp_2=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=2.0)

P_ctm_dgp_3=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=3.0)

P_ctm_dgp_4=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=4.0)

P_ctm_dgp_5=CTM(nk=1024).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=5.0)



## Calc damped power spec at kc=5.0 for DGP, using the CTM code

P_ctm_dgp_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, kc=5.0)

P_ctm_dgp_1_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=1.0, kc=5.0)

P_ctm_dgp_2_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=2.0, kc=5.0)

P_ctm_dgp_3_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=3.0, kc=5.0)

P_ctm_dgp_4_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=4.0, kc=5.0)

P_ctm_dgp_5_k5=CTM(nk=5000).ctm_power(input_k=k_vals, input_A=A_vals, input_z=z_vals, input_P=input_P, input_k_init=input_k, z_val=5.0, kc=5.0)



## Calc undamped power spec for \Lambda CDM using the CTM code

P_ctm=CTM(nk=1024).ctm_power(input_k=k_vals)

P_ctm_1=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=1.0)

P_ctm_2=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=2.0)

P_ctm_3=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=3.0)

P_ctm_4=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=4.0)

P_ctm_5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=5.0)



## Calc damped power spec at kc=5.0 for \Lambda CDM using the CTM code

P_ctm_k5=CTM(nk=1024).ctm_power(input_k=k_vals, kc=5.0)

P_ctm_1_k5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=1.0, kc=5.0)

P_ctm_2_k5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=2.0, kc=5.0)

P_ctm_3_k5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=3.0, kc=5.0)

P_ctm_4_k5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=4.0, kc=5.0)

P_ctm_5_k5=CTM(nk=1024).ctm_power(input_k=k_vals, z_val=5.0, kc=5.0)



## Save the calculated data in .txt file in your directory

np.savetxt(your_path+"dgp_runs/P_dgp_0.txt", P_ctm_dgp)
np.savetxt(your_path+"dgp_runs/P_dgp_1.txt", P_ctm_dgp_1)
np.savetxt(your_path+"dgp_runs/P_dgp_2.txt", P_ctm_dgp_2)
np.savetxt(your_path+"dgp_runs/P_dgp_3.txt", P_ctm_dgp_3)
np.savetxt(your_path+"dgp_runs/P_dgp_4.txt", P_ctm_dgp_4)
np.savetxt(your_path+"dgp_runs/P_dgp_5.txt", P_ctm_dgp_5)

np.savetxt(your_path+"dgp_runs/P_dgp_0_k5.txt", P_ctm_dgp_k5)
np.savetxt(your_path+"dgp_runs/P_dgp_1_k5.txt", P_ctm_dgp_1_k5)
np.savetxt(your_path+"dgp_runs/P_dgp_2_k5.txt", P_ctm_dgp_2_k5)
np.savetxt(your_path+"dgp_runs/P_dgp_3_k5.txt", P_ctm_dgp_3_k5)
np.savetxt(your_path+"dgp_runs/P_dgp_4_k5.txt", P_ctm_dgp_4_k5)
np.savetxt(your_path+"dgp_runs/P_dgp_5_k5.txt", P_ctm_dgp_5_k5)

np.savetxt(your_path+"dgp_runs/P_cdm_0.txt", P_ctm)
np.savetxt(your_path+"dgp_runs/P_cdm_1.txt", P_ctm_1)
np.savetxt(your_path+"dgp_runs/P_cdm_2.txt", P_ctm_2)
np.savetxt(your_path+"dgp_runs/P_cdm_3.txt", P_ctm_3)
np.savetxt(your_path+"dgp_runs/P_cdm_4.txt", P_ctm_4)
np.savetxt(your_path+"dgp_runs/P_cdm_5.txt", P_ctm_5)

np.savetxt(your_path+"dgp_runs/P_cdm_0_k5.txt", P_ctm_k5)
np.savetxt(your_path+"dgp_runs/P_cdm_1_k5.txt", P_ctm_1_k5)
np.savetxt(your_path+"dgp_runs/P_cdm_2_k5.txt", P_ctm_2_k5)
np.savetxt(your_path+"dgp_runs/P_cdm_3_k5.txt", P_ctm_3_k5)
np.savetxt(your_path+"dgp_runs/P_cdm_4_k5.txt", P_ctm_4_k5)
np.savetxt(your_path+"dgp_runs/P_cdm_5_k5.txt", P_ctm_5_k5)
