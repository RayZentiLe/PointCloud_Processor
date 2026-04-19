# PointCloud Processor

PointCloud Processor is a desktop application for loading, viewing, filtering, and exporting point clouds and meshes.

## 1. How to Run

### Requirements
- Conda installed (Miniconda or Anaconda)
- Python 3.11 environment
- Required Python packages: `numpy`, `scipy`, `PySide6`, `vtk`, `open3d`, `PyOpenGL`, `PyOpenGL_accelerate`

### Windows setup
On Windows, use the provided batch files `setup_env.bat` and `run.bat`.

From the repository root:

```cmd
REM Option A: provide your own conda env name
setup_env.bat my_env

REM Option B: let it create "pcprocessor"
setup_env.bat

REM Then run
run.bat pcprocessor
```

### Linux setup
From the repository root:

```bash
./setup_env.sh
```

This creates or activates a Conda environment named `pcprocessor` by default and installs the required packages.

If you want a different environment name:

```bash
./setup_env.sh myenv
```

### Start the application
Once the environment is active:

```bash
./run.sh
```

Or manually:

```bash
conda activate pcprocessor
cd src
python main.py
```

## 2. What this processor can do

### Supported file formats
- Import point clouds: `*.ply`, `*.pcd`, `*.obj`, `*.stl`, `*.xyz`, `*.txt`
- Export point clouds: `*.ply` (binary or ASCII), `*.xyz`, `*.txt`
- Import meshes: `*.ply`, `*.obj`, `*.stl`, `*.off`, `*.gltf`, `*.glb`
- Export meshes: `*.ply` (binary or ASCII), `*.obj`

### Main functions
- Load point clouds and meshes into the workspace
- Export selected layers in supported formats
- Apply PCA planar filtering to a selected point cloud
- Perform Poisson surface reconstruction from a point cloud to a mesh
- Filter a mesh to keep the largest connected component and label small components
- Remove noise from a point cloud using a reference mesh and distance threshold
- Combine two same-type layers into a single merged layer
- Manage multiple layers via the Layers panel
- View logs and task progress

## 3. How to use it (User guideline)

### Open a file
1. Click `Open` in the toolbar or use `File > Open...`
2. Choose a supported point cloud or mesh file.
3. The loaded object appears in the Layers panel and the viewport.

### Select a layer
- Click the desired layer in the Layers panel.
- The selected layer is used by export and filter operations.

### Export a layer
1. Select a layer in the Layers panel.
2. Click `Export` in the toolbar or use `File > Export Selected...`
3. Choose the file format and location.
4. For point clouds, choose `PLY`, `XYZ`, or `TXT`.
5. For meshes, choose `PLY` or `OBJ`.

### PCA Filter (Point Cloud)
1. Select a point cloud layer.
2. Click `PCA Filter` in the toolbar.
3. Set:
   - Search Radius
   - Planarity Threshold
   - Min Neighbors (k)
   - Chunk Size
4. Run the filter to create a new mask group with positive/negative sublayers.

### Poisson Reconstruction
1. Select a point cloud layer.
2. Click `Poisson` in the toolbar.
3. Set reconstruction parameters:
   - Octree Depth
   - Scale
   - Density Quantile Trim
   - Linear Fit
4. Run to generate a new mesh layer.

### Mesh Filter
1. Select a mesh layer.
2. Click `Mesh Filter` in the toolbar.
3. The filter finds connected components and keeps the largest one.
4. The result is stored as positive/negative mask sublayers on the mesh layer.

### Noise Removal
1. Select a point cloud layer.
2. Click `Noise Removal` in the toolbar.
3. Choose a reference mesh and an optional mesh sublayer.
4. Set the distance threshold.
5. Run to mark points as `clean` or `noise`.

### Combine layers
1. Click `Combine` in the toolbar.
2. Choose two layers of the same type (both point clouds or both meshes).
3. Enter a result name.
4. The combined layer is added to the Layers panel.

### Layer management
- Use the Layers panel to rename, delete, and inspect loaded layers.
- Sub-layers created by filters appear under the parent layer.
- Visibility and selection changes are reflected in the viewport.

## Notes
- `*.txt` importer accepts space-separated, comma-separated, or mixed delimiters.
- TXT import supports `x y z` or `x y z r g b` formats.
- Mesh export is currently limited to `PLY` and `OBJ`.
- The application uses Open3D for geometry I/O and PySide6 for the UI.

### GPU vs CPU
- The current code does not include explicit GPU compute acceleration for point cloud processing.
- `numpy` and `scipy` are CPU-based libraries by default.
- `open3d` can support GPU in some of its advanced modules, but this app currently uses standard CPU functions such as `read_point_cloud`, `read_triangle_mesh`, and CPU-based filtering.
- `vtk`/`PySide6` will use the system graphics stack for viewport rendering, but that is rendering acceleration rather than GPU-based point cloud computation.
- Therefore, large files will still be processed on the CPU, and performance may be limited by CPU speed and memory.
- If no GPU is available, the app should still run normally on CPU-only systems.
- If you want GPU acceleration, the code would need to be extended to use Open3D GPU-specific APIs or another GPU-accelerated processing library.

## Developer Notes
- `src/main.py` is the application entry point.
- `src/ui/main_window.py` contains the main window, toolbar actions, file I/O, filters, and task launch logic.
- `src/ui/toolbar.py` defines the main toolbar buttons for open, export, filters, combine, and noise removal.
- `src/ui/dialogs/` holds parameter dialogs for PCA, Poisson reconstruction, mesh filtering, noise removal, and layer combining.
- `src/io_utils/ply_io.py` handles file loading and TXT parsing for point clouds.
- `src/io_utils/exporter.py` handles exporting point clouds and meshes, including TXT export.
- `src/tools/` contains the core filter implementations:
  - `pca_filter.py`
  - `poisson.py`
  - `mesh_filter.py`
  - `noise_removal.py`
- `src/core/layer.py` defines point cloud and mesh layer data structures plus mask groups.
- `src/core/layer_manager.py` manages layer state, selection, visibility, and sublayer masks.
- `src/workers/task_runner.py` runs long tasks in a background thread and reports progress.
