# --------------------------------------------------------------------------- #
#      Pitot: Bringing Runtime Prediction up to speed for Edge Computing      #
# --------------------------------------------------------------------------- #

.phony: dataset

# ----------------------------- Data Processing ----------------------------- #

PRE=python preprocess.py

MF_SESSIONS=$(addprefix data-raw/matrix/, $(shell ls data-raw/matrix))
IF2_SESSIONS=$(addprefix data-raw/if2/, $(shell ls data-raw/if2))
IF3_SESSIONS=$(addprefix data-raw/if3/, $(shell ls data-raw/if3))
IF4_SESSIONS=$(addprefix data-raw/if4/, $(shell ls data-raw/if4))

dataset: data/data.npz data/if2.npz data/if3.npz data/if4.npz

data:
	mkdir -p data

data/data.npz: data data/_platforms.npz data/_opcodes.npz
	$(PRE) dataset \
		-s $(MF_SESSIONS) data-raw/embedded/data.json \
		-c data/_opcodes.npz -p data/_platforms.npz \
		-o data/data.npz
	
data/_opcodes.npz: data
	$(PRE) opcodes --plot -s data-raw/opcodes -o $@

data/_platforms.npz: data
	$(PRE) platforms --plot \
		-s data-raw/matrix/polybench/runtimes.json \
		data-raw/embedded/manifest.json -o $@

data/if2.npz: data/data.npz
	$(PRE) interference -s $(IF2_SESSIONS) -d data/data.npz -o data/if2.npz

data/if3.npz: data/data.npz
	$(PRE) interference -s $(IF3_SESSIONS) -d data/data.npz -o data/if3.npz

data/if4.npz: data/data.npz
	$(PRE) interference -s $(IF4_SESSIONS) -d data/data.npz -o data/if4.npz


# -------------------------- Train/Val/Test Splits -------------------------- #

splits: scripts/split.py
	python manage.py split --seed 42 --replicates 5
