#!/usr/bin/gnuplot
reset

# output definitions
set terminal epslatex size 5.0,5.0 color colortext
set output 'plot_output.tex'
set terminal svg size 500,500 fname 'Verdana' fsize 10
set output 'plot_output.svg'

# labels and titles
set xlabel 'Time (secs)'
set ylabel 'Signal (V)'
set title 'A fancy title here'

# color definitions
set border linewidth 1.5
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 0.7   # --- blue
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 0.7  # --- red

# set legends
set key top right

# set ranges
set xrange [0:180]
set yrange [-1.01:1.01]

plot 'wave_good.dat' title 'Good Data' w linespoints ls 1, \
     'wave.dat' title 'Bad Data' w linespoints ls 2
