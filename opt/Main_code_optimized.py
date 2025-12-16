import numpy as np
import matplotlib.pyplot as plt
from ElementPattern_optimized import ElementPattern
from GenerateLattice_optimized import GenerateLattice
from SubArraySet_Generation_optimized import SubArraySet_Generation

# ANTENNA ARRAY REGULAR CLUSTERING
# Version: 1.0;
# Developed by Milan RC. delivered on 06/30/2017.
# Optimized version with vectorized operations

# __________________________________________________________________________
### INPUT:

# ANTENNA ARRAY PARAMETERS
f = 29e9  # Frequency [GHz]

Nz = 8  # Number of rows
Ny = 8  # Number of columns

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

# OPTIMIZATION 1: Vectorized computation of Yc and Zc
# Pre-compute minimum values for indexing offset
min_NN = np.min(NN)
min_MM = np.min(MM)

# Extract indices for all cluster elements at once
Iy_all = (Cluster[::2] - min_NN).astype(int)  # Even indices (columns 0, 2, 4...)
Iz_all = (Cluster[1::2] - min_MM).astype(int)  # Odd indices (columns 1, 3, 5...)

# Reshape to separate cluster elements and sub-arrays
Ntrans = Cluster.shape[1]
Lsub_elements = B.shape[0]
Iy_all = Iy_all.reshape(Lsub_elements, Ntrans)
Iz_all = Iz_all.reshape(Lsub_elements, Ntrans)

# Use advanced indexing to extract Y and Z coordinates
Yc = Y[Iz_all, Iy_all]
Zc = Z[Iz_all, Iy_all]

# Compute sub-array properties
Lsub = np.full(Ntrans, Lsub_elements, dtype=int)
Zc_m = np.mean(Zc, axis=0)  # Phase center of sub-array (vectorized)
Yc_m = np.mean(Yc, axis=0)  # Phase center of sub-array (vectorized)

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
# OPTIMIZATION 2: Fully vectorized sub-array kernel computation
# Shape: (Lsub_elements, Ntrans) for coordinates
# Shape: (Nv*Nw, 1) for VV and WW flattened

# Flatten VV and WW for broadcasting
VV_flat = VV.flatten()[:, np.newaxis]  # Shape: (Nv*Nw, 1)
WW_flat = WW.flatten()[:, np.newaxis]  # Shape: (Nv*Nw, 1)
Fel_flat = Fel_VW.flatten()[:, np.newaxis]  # Shape: (Nv*Nw, 1)

# Compute phase terms for all elements and sub-arrays at once
# Broadcasting: (Nv*Nw, 1) @ (1, Ntrans) and (Lsub_elements, Ntrans)
# Result shape after summing over elements: (Nv*Nw, Ntrans)
KerFF_sub = np.zeros((Nv * Nw, Ntrans), dtype=complex)

for jj in range(Lsub_elements):
    # Vectorized computation over all sub-arrays and all observation points
    phase_term = np.exp(1j * (VV_flat * Yc[jj, :] + WW_flat * Zc[jj, :]))
    KerFF_sub += phase_term * Fel_flat

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
plt.ylabel("Normalized RPE R(θ,φ)")
plt.legend()
plt.title("Horizontal plane")

plt.show()
