# --------------------------------------------------------------------------- #
#      Pitot: Bringing Runtime Prediction up to speed for Edge Computing      #
# --------------------------------------------------------------------------- #

.phony: typecheck figures dataset

# --------------------------- Dataset Compilation --------------------------- #

MF_SESSIONS=$(addprefix data-raw/matrix/, $(shell ls data-raw/matrix))
IF2_SESSIONS=$(addprefix data-raw/if2/, $(shell ls data-raw/if2))
IF3_SESSIONS=$(addprefix data-raw/if3/, $(shell ls data-raw/if3))
IF4_SESSIONS=$(addprefix data-raw/if4/, $(shell ls data-raw/if4))

dataset: data/data.npz

data:
	mkdir -p data

data/data.npz: data data/_platforms.npz data/_opcodes.npz
	python dataset.py dataset \
		-s $(MF_SESSIONS) data-raw/embedded/data.json \
		-c data/_opcodes.npz -p data/_platforms.npz \
		-o data/data.npz
	
data/_opcodes.npz: data
	python dataset.py opcodes --plot -s data-raw/opcodes -o $@

data/_platforms.npz: data
	python dataset.py platforms --plot \
		-s data-raw/matrix/polybench/runtimes.json \
		data-raw/embedded/manifest.json -o $@

data/if2.npz: data/data.npz
	python dataset.py interference -s $(IF2_SESSIONS) \
		-d data/data.npz -o data/if2.npz

data/if3.npz: data/data.npz
	python dataset.py interference -s $(IF3_SESSIONS) \
		-d data/data.npz -o data/if3.npz

data/if4.npz: data/data.npz
	python dataset.py interference -s $(IF4_SESSIONS) \
		-d data/data.npz -o data/if4.npz

# todo

figures:
	mkdir -p figures
	python plot.py compare
	python plot.py figures
	python plot.py tsne
	python plot.py marginals
	python plot.py simulation


typecheck:
	python -m mypy prediction


simulation:
	python manage.py simulate -j 1000 -s 0.1
	python manage.py simulate -j 900 -s 0.09
	python manage.py simulate -j 800 -s 0.08
	python manage.py simulate -j 700 -s 0.07
	python manage.py simulate -j 600 -s 0.06
	python manage.py simulate -j 500 -s 0.05
	python manage.py simulate -j 400 -s 0.04
	python manage.py simulate -j 300 -s 0.03
	python manage.py simulate -j 200 -s 0.02
	python manage.py simulate -j 100 -s 0.01
