#!/usr/bin/gnuplot
reset

# output definitions
set terminal svg size 800,800 fname 'Verdana' fsize 10
set output 'plot_output_NBYTES_100.svg'

# labels and titles
set xlabel 'Memory Footprint (KByte)'
set ylabel 'Performance (MFlop/s)'
set title 'Data structure comparison: 100 Bytes'

# color definitions
set border linewidth 1.5
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 0.7   # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 0.7  # --- red
set style line 3 lc rgb '#ff00ff' lt 1 lw 2 pt 5 ps 0.7  # --- purple
set style line 4 lc rgb '#00ff00' lt 1 lw 2 pt 5 ps 0.7  # --- green
set style line 5 lc rgb '#663300' lt 1 lw 2 pt 5 ps 0.7  # --- brown
set style line 6 lc rgb '#ffcc00' lt 1 lw 2 pt 5 ps 0.7  # --- yellow

# set legends
set key top right

# set ranges
# set xrange [0:180]
# set yrange [-1.01:1.01]

plot 'temp_int_100_dist.dat' title 'internal - dist' w linespoints ls 1, \
     'temp_int_100_check.dat' title 'internal - check' w linespoints ls 2, \
     'temp_int_100_main.dat' title 'internal - main' w linespoints ls 3, \
     'temp_ext_100_dist.dat' title 'external - dist' w linespoints ls 4, \
     'temp_ext_100_check.dat' title 'external - check' w linespoints ls 5, \
     'temp_ext_100_main.dat' title 'external - main' w linespoints ls 6, \
