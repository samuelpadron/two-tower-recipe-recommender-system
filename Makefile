install:
	uv venv
	. .venv/bin/activate
	# uv python install
	uv pip install --all-extras --requirement pyproject.toml