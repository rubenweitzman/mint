#!/usr/bin/env python3
"""
PPI Prediction Evaluation Script (Profluent-style) for MINT

Evaluates MINT PPI prediction on datasets from dataset_to_eval.md.
Loads MDS datasets from GCS and runs PPI prediction pipeline using MINT.

By default, computes unsupervised scores (cosine similarity of embeddings from concatenated sequences).
With --compute-supervised flag, also computes supervised MLP predictions.

Usage:
    # Default: Only unsupervised scores (embedding similarity)
    python eval_profluent_style/eval_profluent_style.py \
        --dataset-name alignment_skempi \
        --checkpoint-path mint.ckpt \
        --config-path data/esm2_t33_650M_UR50D.json \
        --output-dir ./results/alignment_skempi

    # With supervised predictions (requires MLP checkpoint)
    python eval_profluent_style/eval_profluent_style.py \
        --dataset-name alignment_skempi \
        --checkpoint-path mint.ckpt \
        --config-path data/esm2_t33_650M_UR50D.json \
        --output-dir ./results/alignment_skempi \
        --mlp-checkpoint bernett_mlp.pth \
        --compute-supervised
"""

import os
import sys
import logging
import click
import pickle
import subprocess
import shlex
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
from tqdm import tqdm

# Get the project root (parent of eval_profluent_style folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for MINT imports
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import streaming (will be available if using pixi or if installed)
try:
    from streaming import StreamingDataset
except ImportError:
    logging.error("streaming package not found. Install with: pip install mosaicml-streaming")
    sys.exit(1)

# Import MINT modules
try:
    from mint.helpers.extract import load_config, CSVDataset, CollateFn, MINTWrapper
    from mint.helpers.predict import SimpleMLP
except ImportError as e:
    logging.error(f"Failed to import MINT modules: {e}")
    logging.error("Make sure MINT is installed: pip install -e .")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset paths - supports both MDS and CSV formats
# MDS paths (streaming format)
DATASET_PATHS_MDS = {
    "alignment_skempi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi",
    "alignment_mutational_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_mutational_ppi",
    "alignment_yeast_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_yeast_ppi_combined",
    "alignment_human_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_human_ppi_combined",
    "alignment_intact_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_intact_ppi",
    "validation_high_score_20_species": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/validation_high_score_20_species",
    "alignment_bindinggym_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_bindinggym_combined",
    "alignment_gold_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_gold_combined",
    "human_validation_with_negatives": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/human_validation_with_negatives",
}

# CSV paths (direct CSV format) - for datasets without MDS
DATASET_PATHS_CSV = {
    "alignment_intact_covid": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_intact_covid.csv",
    "alignment_virus_human": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_virus_human.csv",
}

# Combined lookup (prefer MDS, fallback to CSV)
DATASET_PATHS = {**DATASET_PATHS_MDS, **DATASET_PATHS_CSV}


def load_csv_dataset(csv_path: str, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Load CSV dataset from GCS or local path.
    
    Args:
        csv_path: GCS path or local path to CSV file
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        Tuple of (original DataFrame, list of sample dicts)
    """
    logger.info(f"Loading CSV dataset from: {csv_path}")
    
    # Download from GCS if needed
    if csv_path.startswith("gs://"):
        local_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        logger.info(f"Downloading to: {local_csv}")
        cmd = f"gcloud storage cp {shlex.quote(csv_path)} {shlex.quote(local_csv)}"
        subprocess.run(shlex.split(cmd), check=True)
        csv_path = local_csv
    
    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"CSV contains {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Apply max_samples limit
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        logger.info(f"Limited to {max_samples} samples")
    
    # Convert to sample list (same format as MDS)
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples", unit="samples"):
        samples.append({
            'sequence': row.get('sequence', ''),
            'value': float(row.get('value', 0.0)),
            'data_source': row.get('data_source', 'default')
        })
    
    logger.info(f"Loaded {len(samples)} samples")
    return df, samples


def load_mds_dataset(gcs_path: str, max_samples: Optional[int] = None, local_cache_dir: Optional[str] = None) -> List[Dict]:
    """
    Load MDS dataset from GCS.
    
    Args:
        gcs_path: GCS path to MDS dataset
        max_samples: Maximum number of samples to load (None for all)
        local_cache_dir: Optional local directory for caching (auto-generated if None)
    
    Returns:
        List of samples, each with 'sequence' and 'value' fields
    """
    logger.info(f"Loading MDS dataset from: {gcs_path}")
    
    # Use temp directory for caching if not provided
    if local_cache_dir is None:
        local_cache_dir = tempfile.mkdtemp(prefix="mds_cache_")
        logger.info(f"Using temporary cache directory: {local_cache_dir}")
    
    dataset = StreamingDataset(
        remote=gcs_path,
        local=local_cache_dir,
        batch_size=1000,
        shuffle=False,
        num_canonical_nodes=1,
        download_timeout=600,
    )
    
    total_samples = len(dataset)
    logger.info(f"Dataset contains {total_samples} samples")
    
    # Determine how many samples to load
    num_to_load = min(max_samples, total_samples) if max_samples else total_samples
    
    samples = []
    with tqdm(total=num_to_load, desc="Loading samples", unit="samples") as pbar:
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append({
                'sequence': sample.get('sequence', ''),
                'value': float(sample.get('value', 0.0)),
                'data_source': sample.get('data_source', 'default')
            })
            pbar.update(1)
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def extract_protein_pairs(samples: List[Dict]) -> List[Dict]:
    """
    Extract protein pairs from samples.
    
    MINT processes pairs directly (both sequences together), so we simply extract
    all pairs as they appear in the data. Unlike ProteomeLM which processes proteins
    independently and then computes pairwise features, MINT needs pairs together.
    
    Returns:
        List of dicts with 'Protein_Sequence_1' and 'Protein_Sequence_2'
    """
    pairs = []
    
    for i, sample in enumerate(tqdm(samples, desc="Extracting protein pairs", unit="samples")):
        seq = sample['sequence']
        
        # Split by comma (always the separator in these datasets)
        # Use split(',', 1) to split only on first comma in case sequence contains commas
        parts = seq.split(',', 1)
        if len(parts) == 2:
            seq1, seq2 = parts[0].strip(), parts[1].strip()
            if seq1 and seq2:
                pairs.append({
                    'Protein_Sequence_1': seq1,
                    'Protein_Sequence_2': seq2
                })
            else:
                logger.warning(f"Empty sequence in sample {i}")
        else:
            logger.warning(f"Sample {i} does not contain comma-separated pair: {seq[:50]}...")
    
    logger.info(f"Extracted {len(pairs)} protein pairs from {len(samples)} samples")
    
    return pairs


def create_csv_file(pairs: List[Dict], output_path: str) -> str:
    """Create a CSV file from pairs."""
    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)
    logger.info(f"Created CSV file: {output_path} with {len(pairs)} pairs")
    return output_path


def run_ppi_prediction(
    csv_file: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 128,
    crop_len: int = 512,
    mlp_checkpoint_path: Optional[str] = None,
    sep_chains: bool = True,
    compute_supervised: bool = False,
) -> Dict:
    """
    Run MINT PPI prediction pipeline.
    
    Args:
        compute_supervised: If True, compute supervised MLP predictions (requires mlp_checkpoint_path)
    
    Returns:
        Dictionary with results including embeddings, unsupervised scores, and optionally supervised predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    logger.info(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    
    # Create dataset and dataloader
    logger.info(f"Loading pairs from CSV: {csv_file}")
    dataset = CSVDataset(csv_file, 'Protein_Sequence_1', 'Protein_Sequence_2')
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=CollateFn(crop_len), 
        shuffle=False
    )
    
    logger.info(f"Dataset contains {len(dataset)} pairs")
    
    # Initialize MINT wrapper
    logger.info(f"Loading MINT model from checkpoint: {checkpoint_path}")
    wrapper = MINTWrapper(
        cfg, 
        checkpoint_path, 
        sep_chains=sep_chains, 
        device=device
    )
    wrapper.eval()
    
    # Generate embeddings for all pairs
    logger.info("Generating embeddings...")
    all_embeddings = []
    
    with torch.no_grad():
        for batch_idx, (chains, chain_ids) in enumerate(tqdm(loader, desc="Generating embeddings", unit="batch")):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            embeddings = wrapper(chains, chain_ids)
            all_embeddings.append(embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
    
    # Compute unsupervised scores (likelihood scores from concatenated sequences)
    # This is computed by default: cosine similarity between protein embeddings
    logger.info("Computing unsupervised scores (embedding similarity)...")
    unsupervised_scores = None
    
    if sep_chains and all_embeddings.shape[1] == 2560:
        # Split concatenated embeddings back into individual protein embeddings
        emb_dim = all_embeddings.shape[1] // 2
        emb1 = all_embeddings[:, :emb_dim]  # First protein embeddings
        emb2 = all_embeddings[:, emb_dim:]  # Second protein embeddings
        
        # Compute cosine similarity between protein pairs
        # Normalize embeddings
        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
        
        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (emb1_norm * emb2_norm).sum(dim=1)
        unsupervised_scores = cosine_sim.numpy()
        
        logger.info(f"Computed unsupervised scores (cosine similarity) for {len(unsupervised_scores)} pairs")
        logger.info(f"Score range: {unsupervised_scores.min():.4f} - {unsupervised_scores.max():.4f}")
    else:
        logger.warning("Cannot compute unsupervised scores: embeddings not in expected format (need sep_chains=True, 2560-dim)")
        unsupervised_scores = np.zeros(len(dataset))
    
    # Binary PPI prediction using MLP (only if requested and checkpoint provided)
    supervised_predictions = None
    if compute_supervised:
        if mlp_checkpoint_path and os.path.exists(mlp_checkpoint_path):
            logger.info(f"Loading MLP model from: {mlp_checkpoint_path}")
            model = SimpleMLP(input_size=all_embeddings.shape[1])
            mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location=device)
            model.load_state_dict(mlp_checkpoint)
            model.eval()
            model.to(device)
            
            logger.info("Running supervised binary PPI prediction...")
            with torch.no_grad():
                embeddings_tensor = all_embeddings.to(device)
                logits = model(embeddings_tensor)
                supervised_predictions = torch.sigmoid(logits).cpu().numpy().flatten()
            
            logger.info(f"Generated supervised predictions for {len(supervised_predictions)} pairs")
            logger.info(f"Prediction range: {supervised_predictions.min():.4f} - {supervised_predictions.max():.4f}")
        else:
            logger.error("--compute-supervised flag provided but MLP checkpoint not found or not provided")
            logger.error("Please provide --mlp-checkpoint path")
            sys.exit(1)
    elif mlp_checkpoint_path:
        logger.info("MLP checkpoint provided but --compute-supervised flag not set. Skipping supervised prediction.")
        
    return {
        'embeddings': all_embeddings.numpy(),
        'unsupervised_scores': unsupervised_scores,
        'supervised_predictions': supervised_predictions,
        'num_pairs': len(dataset),
    }


@click.command()
@click.option(
    "--dataset-name",
    type=str,
    help="Dataset name from dataset_to_eval.md (e.g., 'alignment_skempi')"
)
@click.option(
    "--gcs-path",
    type=str,
    help="GCS path to MDS dataset (overrides dataset-name if provided)"
)
@click.option(
    "--csv-path",
    type=str,
    help="GCS or local path to CSV file (format: sequence,value,data_source). Overrides --gcs-path"
)
@click.option(
    "--checkpoint-path",
    type=str,
    required=True,
    help="Path to MINT checkpoint (e.g., 'mint.ckpt')"
)
@click.option(
    "--config-path",
    type=str,
    required=True,
    help="Path to model config JSON (e.g., 'data/esm2_t33_650M_UR50D.json')"
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for results"
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to process (for testing, None = all)"
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help="Device for MINT model (default: cuda:0)"
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    help="Batch size for inference (default: 128)"
)
@click.option(
    "--crop-len",
    type=int,
    default=512,
    help="Maximum sequence length (default: 512)"
)
@click.option(
    "--mlp-checkpoint",
    type=str,
    default=None,
    help="Path to MLP checkpoint for supervised binary PPI prediction (required if --compute-supervised)"
)
@click.option(
    "--compute-supervised/--no-compute-supervised",
    default=False,
    help="Compute supervised MLP predictions (requires --mlp-checkpoint). Default: False (only unsupervised scores)"
)
@click.option(
    "--sep-chains/--no-sep-chains",
    default=True,
    help="Use separate chain embeddings (default: True, required for unsupervised scores)"
)
def main(
    dataset_name: Optional[str],
    gcs_path: Optional[str],
    csv_path: Optional[str],
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    max_samples: Optional[int],
    device: str,
    batch_size: int,
    crop_len: int,
    mlp_checkpoint: Optional[str],
    compute_supervised: bool,
    sep_chains: bool,
) -> None:
    """Run MINT PPI prediction evaluation on MDS or CSV dataset."""
    
    # Determine data source (CSV takes priority, then MDS)
    use_csv = False
    original_df = None
    
    if csv_path:
        # Direct CSV path provided
        data_path = csv_path
        use_csv = True
    elif gcs_path:
        # Direct GCS path to MDS
        data_path = gcs_path
        use_csv = gcs_path.endswith('.csv')
    elif dataset_name and dataset_name in DATASET_PATHS:
        data_path = DATASET_PATHS[dataset_name]
        # Check if this dataset is CSV format
        use_csv = dataset_name in DATASET_PATHS_CSV or data_path.endswith('.csv')
    else:
        logger.error(f"Must provide --csv-path, --gcs-path, or --dataset-name (one of: {list(DATASET_PATHS.keys())})")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("MINT PPI Prediction Evaluation")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name or 'custom'}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Format: {'CSV' if use_csv else 'MDS'}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset (CSV or MDS)
    if use_csv:
        logger.info("\n[Step 1/4] Loading CSV dataset...")
        original_df, samples = load_csv_dataset(data_path, max_samples=max_samples)
    else:
        logger.info("\n[Step 1/4] Loading MDS dataset...")
        samples = load_mds_dataset(data_path, max_samples=max_samples)
    
    # Step 2: Extract protein pairs
    logger.info("\n[Step 2/4] Extracting protein pairs...")
    pairs = extract_protein_pairs(samples)
    
    if len(pairs) == 0:
        logger.error("No protein pairs extracted from samples!")
        sys.exit(1)
    
    # Step 3: Create CSV file
    logger.info("\n[Step 3/4] Creating CSV file...")
    csv_file = str(output_path / "protein_pairs.csv")
    create_csv_file(pairs, csv_file)
    
    # Step 4: Run PPI prediction
    logger.info("\n[Step 4/4] Running MINT PPI prediction...")
    results = run_ppi_prediction(
        csv_file=csv_file,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_path,
        device=device,
        batch_size=batch_size,
        crop_len=crop_len,
        mlp_checkpoint_path=mlp_checkpoint,
        sep_chains=sep_chains,
        compute_supervised=compute_supervised,
    )
    
    # Step 5: Create CSV output with predictions
    logger.info("\n[Step 5/5] Creating CSV output with predictions...")
    
    # Unsupervised scores are always computed (default)
    unsupervised_scores = results['unsupervised_scores']
    # Supervised predictions are only computed if flag is set
    supervised_predictions = results['supervised_predictions']
    
    # If we loaded from CSV, add columns to original DataFrame
    if use_csv and original_df is not None:
        logger.info("Adding prediction columns to original CSV...")
        output_df = original_df.copy()
        
        # Add unsupervised scores
        if len(unsupervised_scores) == len(output_df):
            output_df['mint_unsupervised'] = unsupervised_scores
        else:
            logger.warning(f"Score count mismatch: {len(unsupervised_scores)} vs {len(output_df)} rows")
            output_df['mint_unsupervised'] = np.nan
            output_df.loc[:len(unsupervised_scores)-1, 'mint_unsupervised'] = unsupervised_scores
        
        # Add supervised predictions if computed
        if supervised_predictions is not None:
            if len(supervised_predictions) == len(output_df):
                output_df['mint_supervised'] = supervised_predictions
            else:
                output_df['mint_supervised'] = np.nan
                output_df.loc[:len(supervised_predictions)-1, 'mint_supervised'] = supervised_predictions
    else:
        # Build output rows from samples (MDS format)
        output_rows = []
        for i, sample in enumerate(tqdm(samples, desc="Creating output", unit="samples")):
            # Unsupervised score (always present)
            if i < len(unsupervised_scores):
                unsupervised_score = float(unsupervised_scores[i])
            else:
                logger.warning(f"Could not find unsupervised score for sample {i}")
                unsupervised_score = np.nan
            
            # Supervised prediction (only if computed)
            supervised_prediction = None
            if supervised_predictions is not None:
                if i < len(supervised_predictions):
                    supervised_prediction = float(supervised_predictions[i])
                else:
                    logger.warning(f"Could not find supervised prediction for sample {i}")
                    supervised_prediction = np.nan
            
            # Preserve original structure and add scores
            row = {
                'data_source': sample.get('data_source', ''),
                'sequence': sample['sequence'],
                'value': sample['value'],
                'mint_unsupervised': unsupervised_score,
            }
            
            # Add supervised prediction if computed
            if supervised_prediction is not None:
                row['mint_supervised'] = supervised_prediction
            
            output_rows.append(row)
        
        output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    csv_output_file = output_path / "results.csv"
    output_df.to_csv(csv_output_file, index=False)
    logger.info(f"Saved CSV results to {csv_output_file}")
    logger.info(f"CSV contains {len(output_df)} rows with columns: {list(output_df.columns)}")
    
    # Also save pickle for detailed analysis
    results_file = output_path / "ppi_results.pkl"
    logger.info(f"\nSaving detailed results to {results_file}")
    
    # Add metadata to results
    results['dataset_name'] = dataset_name or 'custom'
    results['data_path'] = data_path
    results['num_samples'] = len(samples)
    results['num_pairs'] = len(pairs)
    
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total samples processed: {len(samples)}")
    logger.info(f"Total pairs: {len(pairs)}")
    
    # Log unsupervised scores (always computed)
    if 'mint_unsupervised' in output_df.columns and not output_df['mint_unsupervised'].isna().all():
        logger.info(f"MINT unsupervised score range: {output_df['mint_unsupervised'].min():.4f} - {output_df['mint_unsupervised'].max():.4f}")
    
    # Log supervised predictions (only if computed)
    if 'mint_supervised' in output_df.columns and not output_df['mint_supervised'].isna().all():
        logger.info(f"MINT supervised prediction range: {output_df['mint_supervised'].min():.4f} - {output_df['mint_supervised'].max():.4f}")
    
    logger.info(f"CSV results saved to: {csv_output_file}")
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Upload results to GCS
    gcs_bucket = "profluent-rweitzman"
    method_name = "mint"
    gcs_base_path = f"gs://{gcs_bucket}/baseline_results/{method_name}/{dataset_name or 'custom'}"
    
    logger.info(f"\nUploading results to GCS: {gcs_base_path}")
    try:
        # Upload CSV file
        csv_gcs_path = f"{gcs_base_path}/results.csv"
        logger.info(f"Uploading {csv_output_file} -> {csv_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(csv_output_file))} {shlex.quote(csv_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded CSV to {csv_gcs_path}")
        
        # Upload pickle file (optional, but useful for detailed analysis)
        pkl_gcs_path = f"{gcs_base_path}/ppi_results.pkl"
        logger.info(f"Uploading {results_file} -> {pkl_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(results_file))} {shlex.quote(pkl_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded pickle to {pkl_gcs_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload to GCS: {e}")
        logger.error("Results are still saved locally")
    except Exception as e:
        logger.error(f"Unexpected error uploading to GCS: {e}")
        logger.error("Results are still saved locally")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()


