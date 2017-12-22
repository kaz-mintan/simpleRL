reset

input = sprintf("%s_%s_%s/action_pwm_%s_%s_%s.csv", tf, dir, mode, tf, dir, mode)
set term postscript enhanced eps color
output_dir = sprintf("%s_%s_%s/action_pwm_%s_%s_%s.eps", tf, dir, mode, tf, dir, mode")
set output output_dir

set xlabel 'episode'
set ylabel 'theta'
plot input w l
reset
