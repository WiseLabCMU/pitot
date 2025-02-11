#   __________________      _____ 
#   ___  __ \__(_)_  /________  /_
#   __  /_/ /_  /_  __/  __ \  __/
#   _  ____/_  / / /_ / /_/ / /_  
#   /_/     /_/  \__/ \____/\__/
#   Bringing Runtime Prediction
#   up to speed for Edge Systems
#

PYTHON=python

PRE=$(PYTHON) preprocess.py
RUN_GPU=$(PYTHON) manage.py
RUN_CPU=JAX_PLATFORM_NAME=cpu $(PYTHON) manage.py

# -- Data processing ----------------------------------------------------------

MF_SESSIONS=$(addprefix data-raw/matrix/, $(shell ls data-raw/matrix))
IF2_SESSIONS=$(addprefix data-raw/if2/, $(shell ls data-raw/if2))
IF3_SESSIONS=$(addprefix data-raw/if3/, $(shell ls data-raw/if3))
IF4_SESSIONS=$(addprefix data-raw/if4/, $(shell ls data-raw/if4))

.phony: dataset
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


# -- Run experiments ----------------------------------------------------------

splits: scripts/split.py
	$(RUN_CPU) split --seed 42 --replicates 5

.phony: experiments
experiments: splits
	$(RUN_GPU) train

# -- Evaluation statistics ----------------------------------------------------

RESULTS=$(shell find results -name 'config.json' | xargs dirname)
SUMMARIES=$(RESULTS:results/%=summary/%.npz)

.phony: evaluate
evaluate: $(SUMMARIES) summary/_embeddings.npz

summary/%.npz: results/%
	-$(RUN_CPU) evaluate -p $<

.phony: alias
alias:
	-cd summary/embedding && ln -sf ../pitot.npz 32.npz
	-cd summary/learned && ln -sf ../pitot.npz 4.npz
	-cd summary/interference && ln -sf ../pitot.npz 2.npz
	-cd summary/weight && ln -sf ../pitot.npz 0.5.npz
	-cd summary/features && ln -sf ../pitot.npz all.npz
	-cd summary/components && ln -sf ../pitot.npz full.npz
	-cd summary/conformal && ln -sf ../pitot.npz nonquantile.npz
	-cd summary/baseline && ln -sf ../pitot.npz pitot.npz

summary/_embeddings.npz:
	-$(RUN_CPU) embeddings -p results/pitot -o summary/_embeddings.npz

# -- Figures ------------------------------------------------------------------

FIGURES=ablations_width ablations hyperparameters interference_norm \
	interference margin_full quantile_selection tsne

.phony: figures
figures: $(FIGURES)

$(FIGURES):
	python plot/$@.py
