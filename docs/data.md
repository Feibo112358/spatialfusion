# Data Format

Note: This method requires at least the H&E high-resolution image and X and Y coordinates extracted through segmentation (see the tutorial on extracting embeddings from H&E only). Paired ST data is optional.

The method requires:

- Image (WSI): (n_px_height, n_px_width)
- Coordinates of cells in image space: (n_cells, 2)

If providing paired ST data, the method accepts an AnnData object. 
At minimum, the adata object must contain:

- adata.obsm['spatial']: this should contain the X and Y coordinates of the cell/nuclei centroid in the high-resolution pixel space of the associated WSI.
- adata.X: this should be the cell x gene matrix of raw counts (! this needs to be single-cell resolution data)
- (optional): adata.obs['celltype']: the annotated celltypes (here called 'major_celltype').

SpatialFusion expects preprocessed and aligned data.

