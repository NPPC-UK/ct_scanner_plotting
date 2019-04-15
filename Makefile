plot_all: install
	./venv/bin/python -m ct_plotting.plot_things -d /mnt/mass/max/BR09_CTdata/mnt/mass/scratch/br09_data/ -m BR9_scan_list.csv		

test: install
	./venv/bin/py.test

clean:
	rm *.svg

install: src/ct_plotting/*.py
	./venv/bin/python ./setup.py install > /dev/null

