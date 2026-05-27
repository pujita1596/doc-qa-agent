ingest:
	.venv/bin/python src/ingest.py

serve:
	.venv/bin/uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000 --reload

watch:
	.venv/bin/python src/watcher.py

eval:
	.venv/bin/python evals/eval_pipeline.py

benchmark:
	.venv/bin/python evals/run_benchmarks.py
