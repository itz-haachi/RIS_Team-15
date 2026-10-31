import numpy as np, os
FC = 28e9; C0 = 299_792_458.0; LAM = C0 / FC; K0 = 2*np.pi/LAM
PT_DBM = 40.0
BW = 100e6; NF_DB = 7.0
NOISE_DBM = -174 + 10*np.log10(BW) + NF_DB
import os
GAMMA_TH_DB = float(os.environ.get("RIS_TH", 10.0))
BS = np.array([float(v) for v in os.environ.get("RIS_BS", "-20,6,10").split(",")])
TAG = os.environ.get("RIS_TAG", "")
UE_H = 1.5

UE_X = np.arange(31.0, 50.0, 1.0); UE_Y = np.arange(11.0, 96.0, 1.0)
KIOSK = (37, 43, 55, 58)

RIS_CAND = {
    "R1": ((40.0, -9.8, 6.0), 90.0),
    "R2": ((25.0, -9.8, 6.0), 90.0),
    "R3": ((49.8, 14.0, 6.0), 180.0),
    "R4": ((49.8, 30.0, 6.0), 180.0),
    "R5": ((30.2, 14.0, 6.0), 0.0),
    "R6": ((60.0, -9.8, 6.0), 90.0),
}
YAW_OFFSETS = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
N_SIDES = [32, 64, 96, 128]
D_EL = LAM/2
G_EL = 4*np.pi*D_EL**2/LAM**2
TRACE = dict(max_depth=4, refraction=False, diffraction=True, synthetic_array=True, seed=42)
KPATHS = 16

def ue_grid():
    X, Y = np.meshgrid(UE_X, UE_Y)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, UE_H)], 1)
    k = KIOSK
    valid = ~((pts[:,0] >= k[0]-0.5) & (pts[:,0] <= k[1]+0.5) & (pts[:,1] >= k[2]-0.5) & (pts[:,1] <= k[3]+0.5))
    return pts, valid, X.shape

def ris_frame(name, yaw_off=0.0):
    c, yaw = RIS_CAND[name]; y = np.deg2rad(yaw + yaw_off)
    n = np.array([np.cos(y), np.sin(y), 0.0]); u = np.array([-np.sin(y), np.cos(y), 0.0]); v = np.array([0, 0, 1.0])
    return np.array(c), n, u, v

def ris_elements(name, side, yaw_off=0.0):
    c, n, u, v = ris_frame(name, yaw_off)
    idx = (np.arange(side) - (side-1)/2) * D_EL
    I, J = np.meshgrid(idx, idx)
    return c + I.ravel()[:, None]*u + J.ravel()[:, None]*v, n
