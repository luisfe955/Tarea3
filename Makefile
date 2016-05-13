graficas.png : MarkovChainMonteCarlo.py datos.dat
	python MarkovChainMonteCarlo.py

datos.dat : stats.py output_*.dat
	python stats.py

output_*.dat : main.x
	time ./main.x 450 0.1

main.x : main.c evolve.c inicial.c
	gcc main.c evolve.c inicial.c -lm -fopenmp -o main.x
