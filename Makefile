plot_all: src/ct_plotting/*.py
	./venv/bin/python -m ct_plotting.plot_things	

clean:
	rm *.svg

