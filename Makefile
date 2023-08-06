.phony: typecheck figures dataset

figures:
	mkdir -p figures
	python manage.py compare
	python manage.py figures
	python manage.py tsne
	python manage.py plot_simulation
	python manage.py embedded

typecheck:
	python -m mypy prediction

dataset: data/data.npz

data/data.npz: data/platforms.npz data/opcodes.npz data/matrix.npz
	python manage.py merge \
		-m data/matrix.npz -p data/opcodes.npz -t data/platforms.npz \
		-o data/data.npz

data/matrix.npz:
	python manage.py matrix -p \
		data-raw/matrix data-raw/matrix.add0 \
		data-raw/matrix.add1/* data-raw/data-embedded/cortex-m7.json \
		-o data/matrix --plot --filter -c 25
	
data/opcodes.npz: data/matrix.npz
	python manage.py opcodes -p data-raw/data-opcodes/ \
		-o data/opcodes -d -m data/matrix.npz

data/platforms.npz: data/matrix.npz
	python manage.py platforms -m data/matrix.npz \
		-p data-raw/matrix/runtimes.json data-raw/matrix.add0/runtimes.json \
		data-raw/data-embedded/embedded.json \
		-o data/platforms -d
