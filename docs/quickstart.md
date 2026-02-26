# Quick Start

A minimal example showing how to embed a dataset using the pretrained AE and GCN:

```python
from spatialfusion.embed.embed import AEInputs, run_full_embedding
import pandas as pd
import scanpy as sc

# Load external embeddings (UNI + scGPT)
uni_df = pd.read_parquet('UNI.parquet')
scgpt_df = pd.read_parquet('scGPT.parquet')

# Load AnnData object
adata = sc.read_h5ad("object.h5ad")

# Mapping sample_name -> AEInputs
sample_name = 'sample1'
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
    device="cuda:0", # if cpu, "cpu"
    combine_mode="average",
    spatial_key='spatial',
    celltype_key='major_celltype',
    save_ae_dir=None,  # optional
)
```

This produces a DataFrame containing the final integrated embedding for all cells/nuclei.
