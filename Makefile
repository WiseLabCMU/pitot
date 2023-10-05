# --------------------------------------------------------------------------- #
#      Pitot: Bringing Runtime Prediction up to speed for Edge Computing      #
# --------------------------------------------------------------------------- #

.phony: typecheck figures dataset

# --------------------------- Dataset Compilation --------------------------- #

MF_SESSIONS=$(addprefix data-raw/matrix/, $(shell ls data-raw/matrix))

dataset: data/data.npz

data:
	mkdir -p data

data/data.npz: data data/platforms.npz data/opcodes.npz data/matrix.npz
	python manage.py merge \
		-m data/matrix.npz -p data/opcodes.npz -t data/platforms.npz \
		-o data/data.npz

data/matrix.npz: $(MF_SESSIONS)
	python manage.py matrix -p \
		$(MF_SESSIONS) data-raw/embedded/data.json \
		-o data/matrix --plot --filter --mincount 25
	
data/opcodes.npz: data/matrix.npz
	python manage.py opcodes -p data-raw/opcodes/ \
		-o data/opcodes -d -m data/matrix.npz

data/platforms.npz: data/matrix.npz
	python manage.py platforms -m data/matrix.npz \
		-p data-raw/matrix/polybench/runtimes.json \
		data-raw/embedded/manifest.json \
		-o data/platforms -d

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
