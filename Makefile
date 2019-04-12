plot_all: install
	./venv/bin/python -m ct_plotting.plot_things	

test: install
	./venv/bin/py.test

clean:
	rm *.svg

install: src/ct_plotting/*.py
	./venv/bin/python ./setup.py install > /dev/null

