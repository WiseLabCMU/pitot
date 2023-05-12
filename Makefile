.phony: typecheck figures

figures:
	mkdir -p figures
	python manage.py compare
	python manage.py figures
	python manage.py tsne
	python manage.py plot_simulation
	python manage.py embedded

typecheck:
	python -m mypy prediction
