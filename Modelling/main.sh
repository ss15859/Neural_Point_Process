#!/bin/bash
for j in 2.0 2.5 3.0
do
	for i in 600 1200 1800 2400 3000 3600 4200
	do
   		python Amatricerun.py $i $j
	done
	python generate_plots.py $j
done


