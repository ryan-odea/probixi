; Tiny 2-panel test geometry for CXI multi-panel intake.
; Data array is 4-D (N, panel, ss, fs); the panel axis (dim1) is selected per
; panel. The two 64x32 panels tile a 64x64 data-space image side by side.
clen = 0.1
photon_energy = 12398.0
res = 13333.3
max_adu = 65535

data = /entry/data/data
dim0 = %
dim2 = ss
dim3 = fs

mask = /entry/data/mask
mask_good = 0x0
mask_bad = 0xFFFFFFFF

0/min_fs = 0
0/max_fs = 31
0/min_ss = 0
0/max_ss = 63
0/corner_x = -31.5
0/corner_y = -31.5
0/fs = +1.0x +0.0y
0/ss = +0.0x +1.0y
0/dim1 = 0

1/min_fs = 32
1/max_fs = 63
1/min_ss = 0
1/max_ss = 63
1/corner_x = 0.5
1/corner_y = -31.5
1/fs = +1.0x +0.0y
1/ss = +0.0x +1.0y
1/dim1 = 1
