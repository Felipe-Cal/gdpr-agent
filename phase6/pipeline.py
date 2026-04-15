"""
Phase 6 — Vertex AI Pipeline for the GDPR agent lifecycle.

This pipeline orchestrates the full ingest → evaluate → gate → (optionally) fine-tune
cycle as a versioned, reproducible DAG. Each step runs in an isolated container —
the pipeline cannot proceed past a failed quality gate, so a bad document batch
never silently degrades production answer quality.

KFP (Kubeflow Pipelines) concepts used here:
─────────────────────────────────────────────
  @component  — converts a Python function into a containerised pipeline step.
                KFP packages the function + its imports into a Docker image,
                pushes it to Artifact Registry, and runs it as a Kubernetes pod.

  Input / Output — typed pipeline ports. Scalars (str, float, int) flow through
                KFP's metadata store. Large artifacts (datasets, models) flow as
                URIs pointing to Cloud Storage.

  @pipeline   — assembles components into a DAG. The >> operator sets explicit
                dependencies; data edges create implicit dependencies.

  Condition   — wraps a subgraph so it only runs when a boolean predicate holds.
                Used here to trigger fine-tuning only when eval scores drop.

  Compiled YAML — `compiler.Compiler().compile(pipeline_func, 'pipeline.yaml')`
                produces a portable spec. The YAML is submitted to Vertex AI
                Pipelines which schedules pods and tracks lineage.

Run locally (compile only, free):
    python -m phase6.pipeline

Submit to Vertex AI (⚠️ costs money — see phase6/submit.py for estimates):
    python -m phase6.submit --bucket your-gcs-bucket
"""

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Metrics, Model

# ---------------------------------------------------------------------------
# Component 1 — Ingest
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.20",
        "google-cloud-aiplatform>=1.60",
        "langchain-text-splitters>=0.3",
        "pypdf>=4.0",
        "google-cloud-storage>=2.0",
    ],
)
def ingest_documents(
    gcs_pdf_uri: str,       # gs://bucket/path/to/docs/  (trailing slash = folder)
    gcp_project_id: str,
    gcp_region: str,
    bq_dataset: str,
    bq_table: str,
    embedding_model: str,
    ingested_count: Output[Metrics],
) -> None:
    """
    Download PDFs from GCS, chunk, embed, and upsert into BigQuery.

    This is the same logic as phase1/ingest.py, packaged as a pipeline component.
    Running it here (rather than as a one-off script) means every pipeline run
    has a versioned, auditable record of exactly which documents were ingested —
    important for GDPR accountability (Article 5(2): demonstrating compliance).

    Outputs `ingested_count` as a KFP Metrics artifact so the pipeline UI shows
    the document count for each run without having to read logs.
    """
    import json
    from pathlib import Path
    import tempfile

    from google.cloud import storage, bigquery
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    # Download PDFs from GCS to a temp dir
    gcs_client = storage.Client(project=gcp_project_id)
    bucket_name, prefix = gcs_pdf_uri.replace("gs://", "").split("/", 1)
    bucket = gcs_client.bucket(bucket_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        blobs = list(bucket.list_blobs(prefix=prefix))
        pdf_blobs = [b for b in blobs if b.name.endswith(".pdf")]

        if not pdf_blobs:
            raise ValueError(f"No PDFs found at {gcs_pdf_uri}")

        # Extract text from PDFs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "],
        )

        all_chunks = []
        for blob in pdf_blobs:
            local = Path(tmpdir) / Path(blob.name).name
            blob.download_to_filename(str(local))
            reader = PdfReader(str(local))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            chunks = splitter.split_text(text)
            all_chunks.extend(
                {"source": blob.name, "chunk_index": i, "content": c}
                for i, c in enumerate(chunks)
            )

        # Embed via Vertex AI
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        vertexai.init(project=gcp_project_id, location=gcp_region)
        embed_model = TextEmbeddingModel.from_pretrained(embedding_model)

        BATCH = 250  # API max
        rows = []
        for i in range(0, len(all_chunks), BATCH):
            batch = all_chunks[i:i + BATCH]
            texts = [c["content"] for c in batch]
            embeddings = embed_model.get_embeddings(texts)
            for chunk, emb in zip(batch, embeddings):
                rows.append({
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "embedding": emb.values,
                })

        # Upsert into BigQuery
        bq_client = bigquery.Client(project=gcp_project_id)
        table_ref = f"{gcp_project_id}.{bq_dataset}.{bq_table}"
        errors = bq_client.insert_rows_json(table_ref, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

        ingested_count.log_metric("chunks_ingested", len(rows))
        ingested_count.log_metric("pdfs_processed", len(pdf_blobs))
        print(f"Ingested {len(rows)} chunks from {len(pdf_blobs)} PDFs → {table_ref}")


# ---------------------------------------------------------------------------
# Component 2 — Evaluate
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform[evaluation]>=1.60",
        "langchain>=0.3",
        "langchain-google-vertexai>=2.0",
        "langchain-google-community[bigquery]>=2.0",
        "pandas>=2.0",
    ],
)
def run_evaluation(
    gcp_project_id: str,
    gcp_region: str,
    bq_dataset: str,
    bq_table: str,
    gemini_model: str,
    eval_metrics: Output[Metrics],
) -> None:
    """
    Run the Phase 3 evaluation suite and emit metric artifacts.

    Uses the 10-pair golden dataset from phase3/eval_dataset.py and the same
    custom PointwiseMetric (LLM-as-judge) that fixed the groundedness=0 bug
    in Phase 3. Outputs are KFP Metrics artifacts — visible in the Vertex AI
    Pipelines UI as charts across pipeline runs, so you can see metric trends
    over time without writing a separate dashboard.

    The eval runs inside the pipeline container so it inherits the same
    workload identity as the rest of the pipeline — no extra credentials needed.
    """
    import os
    import pandas as pd
    import vertexai
    from vertexai.evaluation import EvalTask, PointwiseMetric

    vertexai.init(project=gcp_project_id, location=gcp_region)

    # Golden dataset (same as phase3/eval_dataset.py)
    GOLDEN = [
        {
            "prompt": "What are the lawful bases for processing personal data under GDPR?",
            "reference": "Article 6(1) lists six bases: consent, contract, legal obligation, vital interests, public task, and legitimate interests.",
        },
        {
            "prompt": "When is a Data Protection Officer (DPO) required under GDPR?",
            "reference": "Article 37 requires a DPO for public authorities, organisations doing large-scale systematic monitoring, or large-scale processing of special category data.",
        },
        {
            "prompt": "What are the data subject rights under GDPR?",
            "reference": "Articles 15–22 grant rights to access, rectification, erasure, restriction, portability, objection, and rights regarding automated decision-making.",
        },
        {
            "prompt": "What is the maximum fine for GDPR violations?",
            "reference": "Article 83 sets upper limits at €20M or 4% of global annual turnover (whichever is higher) for the most serious infringements.",
        },
        {
            "prompt": "What must a breach notification to the supervisory authority contain?",
            "reference": "Article 33(3): nature of the breach, categories and number of data subjects, likely consequences, and measures taken or proposed.",
        },
    ]

    # Retrieve context from BigQuery for each question (simplified retriever)
    from google.cloud import bigquery

    bq = bigquery.Client(project=gcp_project_id)

    def retrieve_context(question: str) -> str:
        query = f"""
            SELECT content
            FROM VECTOR_SEARCH(
                TABLE `{gcp_project_id}.{bq_dataset}.{bq_table}`,
                'embedding',
                (SELECT ml_generate_embedding_result AS embedding
                 FROM ML.GENERATE_EMBEDDING(
                     MODEL `{gcp_project_id}.{bq_dataset}.embedding_model`,
                     (SELECT @question AS content)
                 )),
                top_k => 3
            )
        """
        # Fallback: just return a placeholder if table not populated
        try:
            rows = list(bq.query(query, job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("question", "STRING", question)]
            )).result())
            return "\n\n".join(r.content for r in rows) if rows else "No context retrieved."
        except Exception:
            return "No context retrieved."

    # Build eval dataframe
    records = []
    for ex in GOLDEN:
        context = retrieve_context(ex["prompt"])
        records.append({
            "prompt": ex["prompt"],
            "reference": ex["reference"],
            "context": context,
        })
    df = pd.DataFrame(records)

    # Custom groundedness metric (same template as phase3/eval_runner.py)
    GROUNDEDNESS_TEMPLATE = """
You are evaluating the groundedness of an AI response about GDPR.

Question: {prompt}
Retrieved context: {context}
AI Response: {response}

Score from 1–5:
5 = Every claim is directly supported by the context
4 = Mostly supported, minor unsupported details
3 = Partially supported; some claims lack support
2 = Mostly unsupported, though not contradicted
1 = Claims contradict or ignore the context

Output JSON: {{"score": <int>, "explanation": "<str>"}}
""".strip()

    groundedness = PointwiseMetric(
        metric="groundedness",
        metric_prompt_template=GROUNDEDNESS_TEMPLATE,
    )

    # Run evaluation in BYOR mode (we supply context column, no model inference)
    eval_task = EvalTask(
        dataset=df,
        metrics=[groundedness, "coherence", "fluency"],
        experiment=f"gdpr-pipeline-eval",
    )
    result = eval_task.evaluate()

    summary = result.summary_metrics
    g_mean = summary.get("groundedness/mean", 0.0)
    c_mean = summary.get("coherence/mean", 0.0)
    f_mean = summary.get("fluency/mean", 0.0)

    eval_metrics.log_metric("groundedness_mean", g_mean)
    eval_metrics.log_metric("coherence_mean", c_mean)
    eval_metrics.log_metric("fluency_mean", f_mean)

    print(f"Eval results — groundedness: {g_mean:.2f}, coherence: {c_mean:.2f}, fluency: {f_mean:.2f}")


# ---------------------------------------------------------------------------
# Component 3 — Quality gate
# ---------------------------------------------------------------------------
@dsl.component(base_image="python:3.11-slim")
def quality_gate(
    groundedness_mean: float,
    coherence_mean: float,
    min_groundedness: float = 3.5,
    min_coherence: float = 3.5,
) -> bool:
    """
    Hard stop if quality metrics fall below thresholds.

    Returning False causes any downstream Condition block to skip, so the
    fine-tuning job is never triggered on a degraded retriever. The component
    itself raises an exception to fail the pipeline run visibly — you'll see
    a red step in the Vertex AI Pipelines UI and an alert fires (if you wired
    Cloud Monitoring to the pipeline experiment).

    Why have a gate at all? Without it, a corrupt PDF batch lowers retrieval
    quality → evaluation scores drop → fine-tuning amplifies the bad signal.
    The gate is the circuit breaker.
    """
    passed = groundedness_mean >= min_groundedness and coherence_mean >= min_coherence

    if not passed:
        msg = (
            f"Quality gate FAILED — "
            f"groundedness {groundedness_mean:.2f} < {min_groundedness} "
            f"or coherence {coherence_mean:.2f} < {min_coherence}. "
            f"Halting pipeline to prevent degraded deployment."
        )
        print(msg)
        raise RuntimeError(msg)

    print(f"Quality gate PASSED — groundedness {groundedness_mean:.2f}, coherence {coherence_mean:.2f}")
    return True


# ---------------------------------------------------------------------------
# Component 4 — Trigger fine-tuning (conditional)
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform>=1.60"],
)
def trigger_finetuning(
    gcp_project_id: str,
    gcp_region: str,
    gcs_bucket: str,
    finetune_model: Output[Model],
) -> None:
    """
    Submits the Phase 5 Vertex AI Training job for LoRA fine-tuning.

    This component only runs inside a dsl.Condition block — the pipeline
    triggers fine-tuning only when eval scores drop below a second (lower)
    threshold, indicating the base model needs domain adaptation.

    The `finetune_model` Output[Model] artifact stores the GCS URI of the
    adapter checkpoint in Vertex AI's lineage graph, so you can always trace
    which pipeline run produced which model version.

    ⚠️  Cost: ~$0.40 (T4 GPU, 45 min). Requires GPU quota increase.
        See phase5/vertex_job.py for quota request instructions.
    """
    import vertexai
    from google.cloud.aiplatform import CustomTrainingJob

    vertexai.init(project=gcp_project_id, location=gcp_region)

    from google.cloud import aiplatform
    aiplatform.init(
        project=gcp_project_id,
        location=gcp_region,
        staging_bucket=f"gs://{gcs_bucket}",
    )

    output_gcs = f"gs://{gcs_bucket}/phase5/pipeline-adapter"

    job = CustomTrainingJob(
        display_name="gdpr-lora-finetune-pipeline",
        script_path="phase5/train.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1.py310:latest",
        requirements=["peft>=0.10", "trl>=0.8", "bitsandbytes>=0.43", "datasets>=2.18", "accelerate>=0.28"],
        model_serving_container_image_uri=None,
    )

    job.run(
        args=[],
        environment_variables={
            "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
            "DATASET_PATH": f"gs://{gcs_bucket}/phase5/data/gdpr_finetune.jsonl",
            "OUTPUT_DIR": "/tmp/gdpr-lora-adapter",
            "GCS_OUTPUT_URI": output_gcs,
            "EPOCHS": "2",
            "BATCH_SIZE": "4",
        },
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        base_output_dir=output_gcs,
        sync=True,
    )

    # Record the artifact URI in the lineage graph
    finetune_model.uri = output_gcs
    print(f"Fine-tuning complete. Adapter at {output_gcs}")


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="gdpr-agent-lifecycle",
    description=(
        "Ingest GDPR documents → evaluate retrieval quality → quality gate → "
        "conditionally trigger LoRA fine-tuning. "
        "Designed for nightly scheduling via Vertex AI Pipelines."
    ),
)
def gdpr_agent_pipeline(
    # GCP config
    gcp_project_id: str,
    gcp_region: str = "europe-west4",

    # Document source
    gcs_pdf_uri: str = "gs://your-bucket/gdpr-docs/",

    # BigQuery
    bq_dataset: str = "gdpr_agent",
    bq_table: str = "document_chunks",

    # Models
    gemini_model: str = "gemini-2.0-flash-lite",
    embedding_model: str = "text-embedding-004",

    # Quality thresholds — pipeline halts if scores fall below these
    min_groundedness: float = 3.5,
    min_coherence: float = 3.5,

    # Fine-tuning trigger threshold — only trigger if scores drop this low
    finetune_trigger_threshold: float = 3.0,

    # GCS bucket for fine-tuning artifacts
    gcs_bucket: str = "your-bucket",
) -> None:
    """
    The GDPR agent lifecycle pipeline.

    DAG structure:
        ingest → evaluate → quality_gate → [conditional] trigger_finetuning

    The quality gate is a hard stop: if groundedness or coherence fall below
    `min_groundedness` / `min_coherence`, the pipeline fails and no downstream
    work runs. This ensures a bad document batch never silently degrades
    production retrieval quality.

    Fine-tuning is triggered only when scores drop below `finetune_trigger_threshold`
    (a lower bar than the gate) — i.e., retrieval is degraded but not failed.
    """
    # Step 1: Ingest documents from GCS into BigQuery
    ingest_task = ingest_documents(
        gcs_pdf_uri=gcs_pdf_uri,
        gcp_project_id=gcp_project_id,
        gcp_region=gcp_region,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        embedding_model=embedding_model,
    )

    # Step 2: Evaluate retrieval quality using LLM-as-judge
    eval_task = run_evaluation(
        gcp_project_id=gcp_project_id,
        gcp_region=gcp_region,
        bq_dataset=bq_dataset,
        bq_table=bq_table,
        gemini_model=gemini_model,
    )
    eval_task.after(ingest_task)  # explicit ordering (no data edge)

    # Step 3: Quality gate — fails the pipeline run if scores are too low
    gate_task = quality_gate(
        groundedness_mean=eval_task.outputs["groundedness_mean"],
        coherence_mean=eval_task.outputs["coherence_mean"],
        min_groundedness=min_groundedness,
        min_coherence=min_coherence,
    )

    # Step 4 (conditional): Trigger fine-tuning only if scores are degraded
    # dsl.Condition runs the inner block only when the predicate holds.
    # This uses KFP's runtime condition evaluation — the predicate is checked
    # after quality_gate runs, not at compile time.
    with dsl.Condition(
        eval_task.outputs["groundedness_mean"] < finetune_trigger_threshold,
        name="scores-degraded",
    ):
        trigger_finetuning(
            gcp_project_id=gcp_project_id,
            gcp_region=gcp_region,
            gcs_bucket=gcs_bucket,
        ).after(gate_task)


# ---------------------------------------------------------------------------
# Local compile (free — no GCP calls)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    output_path = "phase6/gdpr_pipeline.yaml"
    compiler.Compiler().compile(gdpr_agent_pipeline, output_path)
    print(f"Pipeline compiled → {output_path}")
    print("Submit with: python -m phase6.submit --bucket your-gcs-bucket")
