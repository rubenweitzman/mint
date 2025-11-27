# Explanation of `eval_profluent_style.py`

## Overview
This script evaluates MINT (Multimeric INteraction Transformer) on protein-protein interaction (PPI) prediction tasks. It loads datasets from Google Cloud Storage (GCS) in MDS (Mosaic Data Shard) format and generates binary PPI predictions.

## What the Script Does (Step-by-Step)

### Step 1: Load MDS Dataset (`load_mds_dataset`)
- **Purpose**: Downloads and loads protein interaction data from GCS
- **Input**: GCS path to MDS dataset (e.g., `gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi`)
- **Process**:
  - Creates a temporary local cache directory
  - Uses `StreamingDataset` to stream data from GCS
  - Extracts samples with fields: `sequence`, `value`, `data_source`
  - `sequence`: Comma-separated pair of protein sequences (e.g., "SEQ1,SEQ2")
  - `value`: Ground truth label/value (e.g., binding affinity, interaction score)
  - `data_source`: Source identifier for the data
- **Output**: List of sample dictionaries

### Step 2: Extract Protein Pairs (`extract_protein_pairs`)
- **Purpose**: Parses comma-separated sequences into separate protein pairs
- **Process**:
  - Splits each `sequence` field by comma (first comma only)
  - Creates pairs with `Protein_Sequence_1` and `Protein_Sequence_2`
  - Handles edge cases (empty sequences, missing commas)
- **Output**: List of dictionaries with two protein sequences

### Step 3: Create CSV File (`create_csv_file`)
- **Purpose**: Converts pairs into CSV format for MINT processing
- **Output**: CSV file with columns: `Protein_Sequence_1`, `Protein_Sequence_2`

### Step 4: Run PPI Prediction (`run_ppi_prediction`)
This is the core prediction pipeline:

#### 4a. Load MINT Model
- Loads model config from JSON (e.g., `esm2_t33_650M_UR50D.json`)
- Initializes `MINTWrapper` with the checkpoint (`mint.ckpt`)
- Sets model to evaluation mode (`wrapper.eval()`)

#### 4b. Generate Embeddings
- **Process**:
  - Creates `CSVDataset` from the pairs CSV
  - Uses `CollateFn` to tokenize and batch sequences (max length: 512)
  - For each batch:
    - Tokenizes sequences with special tokens (`<cls>`, `<eos>`)
    - Processes both proteins together (MINT is designed for pairs)
    - Generates embeddings using MINT's transformer layers
    - If `sep_chains=True` (default): Concatenates embeddings from both chains → shape `[batch_size, 2560]`
    - If `sep_chains=False`: Averages embeddings → shape `[batch_size, 1280]`
- **Output**: Tensor of embeddings `[num_pairs, embedding_dim]`

#### 4c. Binary PPI Prediction (SUPERVISED)
- **Purpose**: Predicts whether two proteins interact (binary classification)
- **Process**:
  - Loads pre-trained MLP classifier (`bernett_mlp.pth`)
  - MLP architecture:
    ```
    Input (2560) → Linear → ReLU → Dropout(0.2) → Linear → Output (1)
    ```
  - Applies MLP to embeddings to get logits
  - Applies sigmoid to get probabilities: `P(interaction) ∈ [0, 1]`
- **Output**: Array of prediction probabilities

### Step 5: Create Output Files
- **CSV Output** (`results.csv`):
  - Preserves original data: `data_source`, `sequence`, `value`
  - Adds `prediction` column with MLP output
- **Pickle Output** (`ppi_results.pkl`):
  - Full embeddings array
  - Predictions array
  - Metadata (dataset name, paths, counts)

## When is "Supervised" Set?

**Important**: There is **NO explicit "supervised" parameter** in this script. However, the script uses **supervised learning** implicitly:

### Supervised Component: MLP Classifier
- The `bernett_mlp.pth` checkpoint is a **supervised classifier** trained on labeled PPI data
- It was trained on the **Bernett et al. gold-standard PPI dataset** (see `mint/downstream/GeneralPPI/ppi/`)
- Training process:
  1. MINT embeddings were extracted for protein pairs
  2. MLP was trained with binary cross-entropy loss on labeled interactions
  3. Labels: 1 = proteins interact, 0 = proteins don't interact
  4. Trained for 30 epochs with AUPRC as the monitoring metric

### Unsupervised Component: MINT Embeddings
- MINT itself is **self-supervised** (trained on unlabeled PPI sequences from STRING database)
- The embeddings capture protein interaction patterns without explicit labels

### Two-Stage Pipeline
```
Input Protein Pairs
    ↓
[MINT (Self-Supervised)] → Embeddings (2560-dim)
    ↓
[MLP (Supervised)] → Binary Predictions (0-1 probability)
```

## Key Parameters

- `--mlp-checkpoint`: **Required** - Path to supervised MLP checkpoint
  - If not provided, script exits with error (line 252-253)
  - This enforces supervised prediction (can't run without it)

- `--sep-chains`: Default `True`
  - `True`: Concatenates embeddings from both proteins → 2560-dim (recommended)
  - `False`: Averages embeddings → 1280-dim

- `--max-samples`: Limits number of samples (for testing)
  - `None`: Process all samples in dataset

## Data Flow Summary

```
GCS MDS Dataset
    ↓ (load_mds_dataset)
Samples: [{sequence: "SEQ1,SEQ2", value: X, data_source: Y}]
    ↓ (extract_protein_pairs)
Pairs: [{Protein_Sequence_1: "SEQ1", Protein_Sequence_2: "SEQ2"}]
    ↓ (create_csv_file)
CSV File
    ↓ (run_ppi_prediction)
MINT Embeddings [N, 2560]
    ↓ (MLP classifier - SUPERVISED)
Predictions [N] (probabilities 0-1)
    ↓ (create output)
results.csv + ppi_results.pkl
```

## Why This Design?

1. **Separation of Concerns**: 
   - MINT provides general-purpose embeddings (can be reused)
   - MLP provides task-specific classification (trained on Bernett dataset)

2. **Flexibility**: 
   - Can use embeddings for other tasks (regression, ranking, etc.)
   - Can train new MLPs for different datasets/tasks

3. **Efficiency**: 
   - Embeddings computed once, can be reused
   - MLP inference is fast (simple forward pass)

