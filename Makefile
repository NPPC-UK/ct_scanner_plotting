plot_all: src/ct_plotting/*.py
	./venv/bin/python -m ct_plotting.plot_things	

test:
	./venv/bin/python ./setup.py install > /dev/null && \
	./venv/bin/py.test

clean:
	rm *.svg

