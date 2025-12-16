import numpy as np
import matplotlib.pyplot as plt
from ElementPattern import ElementPattern
from GenerateLattice import GenerateLattice
from SubArraySet_Generation import SubArraySet_Generation
import time

# ANTENNA ARRAY REGULAR CLUSTERING
# Version: 1.0;
# Developed by Milan RC. delivered on 06/30/2017.

# __________________________________________________________________________
### INPUT:
start = time.time()
# ANTENNA ARRAY PARAMETERS
f = 29e9  # Frequency [GHz]

Nz = 16  # Number of rows
Ny = 16  # Number of columns

dist_z = 0.7  # antenna distance on z axis [times lambda]
dist_y = 0.5  # antenna distance on y axis [times lambda]

azi0 = 0  # [deg] azimuth steering angle
ele0 = 10  # [deg] elevation steering angle

### SINGLE ELEMENT RPE:
P = 1  # Set P=0 to achieve an isotropic element pattern, set P=1 for
# cosine element pattern
load_file = 0  # Set load_file =1 to load antenna element RPE from HFSS,
# set load_file =o to generate isotropic pattern
file_name = "RPE_element.csv"  # name of the file with antenna element RPE from HFSS

### SELECT CLUSTER TYPE: deselect the one you need
# B = np.array([[0, 0]])  # single element cluster / NO clastering solution
### cluster size: 2 antenna elements
B = np.array([[0, 0], [0, 1]])  # vertical linear cluster
# B = np.array([[0, 0], [1, 0]])  # horizontal linear cluster
### cluster size: 3 antenna elements
# B = np.array([[0, 0], [0, 1], [0, 2]])  # vertical linear cluster
# B = np.array([[0, 0], [1, 0], [2, 0]])  # horizontal linear cluster
### cluster size: 4 antenna elements
# B = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])  # vertical linear cluster
# B = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])  # horizontal linear cluster

#############################################################################
#############################
#         LATTICE PARAMETERS
#############################################################################
### LATTICE SELECTION - select the type of lattice
scale = 3e8 / f * 1000  # [mm]
lambda_ = 3e8 / f  # [m]
beta = 2 * np.pi / lambda_

### Rectangular grid
dz = dist_z * lambda_
dy = dist_y * lambda_
x1 = np.array([dy, 0])
x2 = np.array([0, dz])

# Generate the basic lattice of grid-points
Y, Z, NN, MM, Dy, Dz, ArrayMask, I = GenerateLattice(Ny, Nz, x1, x2)

#############################################################################
#                       POLAR COORDINATE
#############################################################################
### AZIMUT AND ELEVATION SAMPLING (for plots)
dele = 0.5  # angle resolution [deg]
dazi = 0.5  # angle resolution [deg]
ele = np.arange(-90, 90 + dele, dele)
azi = np.arange(-90, 90 + dazi, dazi)
AZI, ELE = np.meshgrid(azi, ele)
WW = beta * np.cos(np.radians(90 - ELE))
VV = beta * np.sin(np.radians(90 - ELE)) * np.sin(np.radians(AZI))
Nw = WW.shape[1]
Nv = VV.shape[0]

#############################################################################

###                 CLUSTER EVALUATION
#############################################################################
### MAPPING ALGORITHM - Sub-array definition
Nel = Nz * Ny  # number of array elements
ElementExc = np.ones((Nz, Ny))  # Fixed array tapering [BFN]

Cluster, Nsub = SubArraySet_Generation(B, NN.flatten(), MM.flatten())
Yc = np.zeros((B.shape[0], Cluster.shape[1] // 2))
Zc = np.zeros((B.shape[0], Cluster.shape[1] // 2))
for kk in range(Cluster.shape[1] // 2):
    for l1 in range(B.shape[0]):
        Iy = int(Cluster[l1, 2 * kk] - np.min(NN))
        Iz = int(Cluster[l1, 2 * kk + 1] - np.min(MM))
        Yc[l1, kk] = Y[Iz, Iy]
        Zc[l1, kk] = Z[Iz, Iy]

Ntrans = Yc.shape[1]
Lsub = np.zeros(Ntrans, dtype=int)
Zc_m = np.zeros(Ntrans)
Yc_m = np.zeros(Ntrans)
for kk in range(Ntrans):
    Lsub[kk] = B.shape[0]
    Zc_m[kk] = np.mean(Zc[0 : Lsub[kk], kk])  # Phase center of sub-array
    Yc_m[kk] = np.mean(Yc[0 : Lsub[kk], kk])  # Phase center of sub-array

plt.figure()
plt.plot(Yc, Zc, "s")
plt.grid(True)
plt.xlabel("y [m]")
plt.ylabel("z [m]")
plt.title("Antenna Sub-arrays")

### EXCITATIONS
v0 = beta * np.sin(np.radians(90 - ele0)) * np.sin(np.radians(azi0))
w0 = beta * np.cos(np.radians(90 - ele0))
Phase_m = np.exp(-1j * (w0 * Zc_m + v0 * Yc_m))  # Clustered phase distribution
Amplit_m = (
    np.ones(Ntrans) / Lsub
)  # Clustered amplitude distribution (normalized to cluster size)
c0 = Amplit_m * Phase_m

#############################################################################
#####################################
#       RPE and ARRAY FACTOR
#############################################################################

### ELEMENT FACTOR - single element radiation pattern
Fel = ElementPattern(P, ELE, AZI, load_file, file_name)
Fel_VW = Fel  # In MATLAB code it's interp2(AZI,ELE,Fel,AZI,ELE) which returns Fel

plt.figure()
plt.contourf(azi, ele, 20 * np.log10(Fel_VW))
plt.xlabel("azimuth φ [deg]")
plt.ylabel("elevation θ [deg]")
plt.title("Radiation Pattern R(θ,φ)")
plt.colorbar(label="[dB]")

### FAR FIELD TRANSFORMATION KERNELS
# sub-array kernel (Sub-Array Radiation pattern)
KerFF_sub = np.zeros(
    (Nw * Nv, Ntrans), dtype=complex
)  # one coefficient for every antenna element (but equal for each sub-array elements)
for kk in range(Ntrans):
    for jj in range(Lsub[kk]):
        KerFF_sub[:, kk] = (
            KerFF_sub[:, kk]
            + np.exp(1j * (VV.flatten() * Yc[jj, kk] + WW.flatten() * Zc[jj, kk]))
            * Fel_VW.flatten()
        )

FF = KerFF_sub @ c0
FF_norm = FF / np.max(np.abs(FF))
FF_norm_2D = FF_norm.reshape(Nv, Nw)
Fopt_dB = 20 * np.log10(np.abs(FF_norm_2D))  # RPE

#############################################################################
#############################
#          PLOTS
#############################################################################

# Figure plotting Normalized Radiation Pattern in the visible range
plt.figure()
levels = np.arange(-50, 2, 2)
plt.contourf(AZI, ELE, Fopt_dB, levels=levels)
plt.xlabel("azimuth φ [deg]")
plt.ylabel("elevation θ [deg]")
plt.title("Normalized Radiation Pattern R(θ,φ)")
plt.colorbar(label="[dB]")

# Figure plotting cardinal planes FF cuts
Iele = np.where((ele - ele0) >= 0)[0][0]
Iazi = np.where((azi - azi0) >= 0)[0][0]
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ele, Fopt_dB[:, Iazi], "b", linewidth=2, label="RPE regular clustered antenna")
plt.plot(ele[Iele] * np.ones(11), np.arange(0, -55, -5), "+r", label="steering angle")
plt.axis([-90, 90, -50, 0])
plt.grid(True)
plt.xlabel("θ")
plt.ylabel("Normalized RPE R(θ,φ) [dB]")
plt.legend()
plt.title("Vertical plane")

plt.subplot(2, 1, 2)
plt.plot(azi, Fopt_dB[Iele, :], "b", linewidth=2, label="RPE regular clustered antenna")
plt.plot(azi[Iazi] * np.ones(11), np.arange(0, -55, -5), "+r", label="steering angle")
plt.axis([-90, 90, -50, 0])
plt.grid(True)
plt.xlabel("φ")
plt.ylabel("Normalized RPE R(θ,φ) [dB]")
plt.legend()
plt.title("Horizontal plane")

end = time.time()
print(f"Tempo di esecuzione: {end - start:.6f} secondi")
plt.show(block=False)
