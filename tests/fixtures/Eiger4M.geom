clen = 0.2007
photon_energy   = 12398.0 ; 12398 / (/entry/instrument/beam/incident_wavelength)


adu_per_eV      = 0.00008066 ; 1 / photon_energy
; adu_per_photon = 1 ; in future versions, you can specify like this
res             = 13333.3 ; 1 m / 75 um
max_adu         = 12287.0 ; /entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff

; used by geoptimiser
;rigid_group_0 = 0
;rigid_group_collection_0 = 0
rigid_group_collection_c0 = g0
rigid_group_g0 = 0

data = /entry/data/data
dim0 = %
dim1 = ss
dim2 = fs

; Uncomment these lines if you have a separate bad pixel map (recommended!)
mask_file = /das/work/units/LBR-FEL/p17489/PROCESS/2018-07-22_SLS_TR-SMX/processing/process_hadamard/mask.h5
mask = /pixel_mask
mask_good = 0x0
mask_bad = 0xFFFFFFFF

; corner_{x,y} set the position of the corner of the detector (in pixels)
; relative to the beam

0/min_fs = 0
0/min_ss = 0
0/max_fs = 2069
0/max_ss = 2166
0/corner_x = -1020.401102
0/corner_y = -1080.459892
0/fs = +1.000000x -0.000y
0/ss = +0.000x +1.000000y
;0/fs            = x
;0/ss            = y

bad_shadow/min_ss = 1032
bad_shadow/max_ss = 1128
bad_shadow/min_fs = 0
bad_shadow/max_fs = 1075

bad_shadow_2/min_ss = 1000
bad_shadow_2/max_ss = 1150
bad_shadow_2/min_fs = 950
bad_shadow_2/max_fs = 1100
