# Quick Start

A minimal example showing how to embed a dataset using the pretrained AE and GCN:

```python
from spatialfusion.embed.embed import AEInputs, run_full_embedding
import pandas as pd
import pathlib as pl

# Load external embeddings (UNI + scGPT)
uni_df = pd.read_parquet('UNI.parquet')
scgpt_df = pd.read_parquet('scGPT.parquet')

# Paths to pretrained models
ae_model_dir = pl.Path('../data/checkpoint_dir_ae/')
gcn_model_dir = pl.Path('../data/checkpoint_dir_gcn/')

# Mapping sample_name -> AEInputs
ae_inputs_by_sample = {
    sample_name: AEInputs(
        adata=adata,
        z_uni=uni_df,
        z_scgpt=scgpt_df,
    ),
}

# Run the multimodal embedding pipeline
emb_df = run_full_embedding(
    ae_inputs_by_sample=ae_inputs_by_sample,
    ae_model_path=ae_model_dir / "spatialfusion-multimodal-ae.pt",
    gcn_model_path=gcn_model_dir / "spatialfusion-full-gcn.pt",
    device="cuda:0",
    combine_mode="average",
    spatial_key='spatial',
    celltype_key='major_celltype',
    save_ae_dir=None,  # optional
)
```

This produces a DataFrame containing the final integrated embedding for all cells/nuclei.
