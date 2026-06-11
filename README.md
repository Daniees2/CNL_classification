# LLM Fine-tuning for CNL Classification (Public Package)

This directory is the upload-ready public version of the project.

## Included

- `project.ipynb` — runnable notebook that includes:
  - a fast fake-data demo,
  - in-notebook regeneration of production figures, and
  - one contained optional cell for full real Qwen fine-tuning runs
- `project.html` — executed HTML export of the notebook
- `data/fake_cnl_sample.csv` — hand-created sample data (shareable)
- `requirements.txt` — **exact pinned package versions from the current environment** (`pip freeze`)
- `models/finetune_base/` — LoRA adapter bundle (real-data fine-tune)
- `models/finetune_synthetic_equal/` — LoRA adapter bundle (uniform synthetic)
- `models/finetune_synthetic_balance/` — LoRA adapter bundle (balanced synthetic)
- `plots/` — exact production PNG figures used in poster/presentation
- `plot_stats/` — companion metrics/statistics JSON files and packaged prediction inputs for in-notebook reproduction
- `scripts/` — exact plotting scripts from the main project used to generate production figures

Each `models/*` folder contains the adapter configuration package (for example `adapter_config.json`, tokenizer/config sidecars, and run metadata) plus split compressed adapter-weight chunks:

- `adapter_model.safetensors.gz.part01`, `part02`, ...
- `adapter_model.safetensors.gz.sha256`

## Not Included

- Real/restricted interview data
- Full checkpoint trees from training (epoch checkpoints, snapshots)
- Private preprocessing artifacts
- Base model weights

## Important Model Clarification

The included fine-tuned model artifacts are **LoRA adapters**, not standalone full models.

To run true Qwen inference with these adapters, you must provide the base model:

- `Qwen2.5-1.5B-Instruct`

Then load:

- Base model: `Qwen2.5-1.5B-Instruct`
- Adapter path (choose one):
  - `models/finetune_base/` for real-data fine-tune
  - `models/finetune_synthetic_equal/` for uniform synthetic augmentation
  - `models/finetune_synthetic_balance/` for class-balanced synthetic augmentation

Adapter wiring notes:

- `adapter_config.json` is included for each adapter variant.
- `adapter_model.safetensors` is stored as split compressed chunks to satisfy GitHub upload limits.

Rebuild adapter weights after cloning (PowerShell, run from repository root):

```powershell
$models = @(
  "models/finetune_base",
  "models/finetune_synthetic_equal",
  "models/finetune_synthetic_balance"
)

foreach ($m in $models) {
  $dir = Resolve-Path $m
  $parts = Get-ChildItem "$dir/adapter_model.safetensors.gz.part*" | Sort-Object Name
  $gz = Join-Path $dir "adapter_model.safetensors.gz"
  $out = Join-Path $dir "adapter_model.safetensors"

  # Join split parts
  $fs = [System.IO.File]::Create($gz)
  foreach ($p in $parts) {
    $bytes = [System.IO.File]::ReadAllBytes($p.FullName)
    $fs.Write($bytes, 0, $bytes.Length)
  }
  $fs.Close()

  # Decompress .gz -> .safetensors
  python -c "import gzip,shutil,sys; gz=sys.argv[1]; out=sys.argv[2]; fi=gzip.open(gz,'rb'); fo=open(out,'wb'); shutil.copyfileobj(fi,fo); fi.close(); fo.close()" "$gz" "$out"
}
```

The notebook includes a lightweight surrogate section for fast execution, and separately displays exact production plots.

## Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace project.ipynb
jupyter nbconvert --to html project.ipynb --output project.html
```

## Notebook flow

1. Run the top real-pipeline cell with `RUN_REAL_PIPELINE=False` for a quick pass.
2. Set `RUN_REAL_PIPELINE=True` to run full Qwen fine-tuning.
3. Set `RUN_EVAL_AND_HOOK_PLOTS=True` in that same cell to refresh `plot_stats/` from new benchmark outputs.
4. Run the production-plot reproduction section to regenerate final figures into `plots/`.

## Exact plot reproduction inside notebook

The notebook contains a section named:

- `Reproduce production plots in-notebook`

That section contains the plotting code directly in notebook cells (no subprocess/script runner).  
It reads packaged inputs from `plot_stats/` and regenerates figures into `plots/`.

There is only one production-figure section in the notebook (no static duplicate section).

## Full real training cell (optional)

The notebook includes one contained code cell for the real training pipeline:

- uses `run_finetune.py` (same project entrypoint as the original workflow)
- uses base model `Qwen/Qwen2.5-1.5B-Instruct` by default
- runs base fine-tune, uniform synthetic fine-tune, and balanced synthetic fine-tune
- includes inline comments showing how to swap to your own local base-model path

This cell is off by default with `RUN_REAL_PIPELINE = False` because it performs real training.

Outputs from that top cell:

- adapters in `../outputs/finetune_base`, `../outputs/finetune_synthetic_equal`, `../outputs/finetune_synthetic_balance`
- optional benchmark outputs in `../outputs/eval_benchmark` when `RUN_EVAL_AND_HOOK_PLOTS=True`
- automatic plot input refresh into `plot_stats/` (same switch), so the production-plot cells regenerate from fresh run results

This reproduces the same figure code paths used in the project for:
- depth accuracy
- categorical heterogeneity
- validation metrics
- error locality
- support diagnostics
- abstractness/accuracy heatmap

## What the demo shows

- Hierarchical classification workflow (Domain -> Component -> Item)
- Baseline vs fine-tuned surrogate comparison
- Validation metrics (accuracy, precision, recall, F1) by depth
- Category heterogeneity, error-locality, and abstractness diagnostics
- Exact production plots and stats from the real pipeline
- Plot styling aligned to poster/presentation conventions
