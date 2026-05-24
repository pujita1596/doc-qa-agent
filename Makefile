ingest:
	.venv/bin/python src/ingest.py

serve:
	.venv/bin/uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000 --reload

eval:
	.venv/bin/python evals/eval_pipeline.py
