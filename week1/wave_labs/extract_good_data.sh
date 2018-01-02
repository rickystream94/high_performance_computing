awk 'NR<=100{print $1, $2}; NR>100{print $1, $3}' wave.dat > wave_good.dat
