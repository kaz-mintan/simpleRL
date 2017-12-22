#!/bin/sh

a = ('happy','sad')

for tf in 'happy', 'sad'
do
	for dir in 'posi','nega'
	do
		for mode in 'heuristic', 'delta'
		do
			gnuplot -e "tf=${tf}, dir=${dir},mode=${mode}" plot.plt
		done
	done
done
