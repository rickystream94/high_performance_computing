# --- NBYTES = 100 ---
# INTERNAL: memory vs mf_dist
awk '{print $1, $2}' internal_NBYTES_100.dat > temp_int_100_dist.dat
# INTERNAL: memory vs mf_check
awk '{print $1, $3}' internal_NBYTES_100.dat > temp_int_100_check.dat
# INTERNAL: memory vs mf_main
awk '{print $1, $4}' internal_NBYTES_100.dat > temp_int_100_main.dat

# EXTERNAL: memory vs mf_dist
awk '{print $1, $2}' external_NBYTES_100.dat > temp_ext_100_dist.dat
# EXTERNAL: memory vs mf_check
awk '{print $1, $3}' external_NBYTES_100.dat > temp_ext_100_check.dat
# EXTERNAL: memory vs mf_main
awk '{print $1, $4}' external_NBYTES_100.dat > temp_ext_100_main.dat

# --- NBYTES = 200 ---
# INTERNAL: memory vs mf_dist
awk '{print $1, $2}' internal_NBYTES_200.dat > temp_int_200_dist.dat
# INTERNAL: memory vs mf_check
awk '{print $1, $3}' internal_NBYTES_200.dat > temp_int_200_check.dat
# INTERNAL: memory vs mf_main
awk '{print $1, $4}' internal_NBYTES_200.dat > temp_int_200_main.dat

# EXTERNAL: memory vs mf_dist
awk '{print $1, $2}' external_NBYTES_200.dat > temp_ext_200_dist.dat
# EXTERNAL: memory vs mf_check
awk '{print $1, $3}' external_NBYTES_200.dat > temp_ext_200_check.dat
# EXTERNAL: memory vs mf_main
awk '{print $1, $4}' external_NBYTES_200.dat > temp_ext_200_main.dat
