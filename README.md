# GenCoal

GenCoal is an open-source software package in Python to generate coal molecular structures.

<img src="docs/source/logos/flowchart.jpg" alt="煤分子结构示例" width="1000">

## Dependencies

To ensure the GenCoal software runs correctly, you will need to install the required dependencies. You can use the following steps to set up the environment using `conda` and `pip`.

### Step 1: Install Conda

If you haven't installed `conda` yet, download and install it from the [official Anaconda website](https://www.anaconda.com/products/distribution) or use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a lighter version.

### Step 2: Clone the Repository

Clone the GenCoal repository to your local machine:

```bash
git clone https://github.com/GYLiGroup/GenCoal.git
cd GenCoal
```

### Step 3: Create a Conda Environment

Use `conda` to create and activate a new environment for GenCoal:

```bash
conda create --name gencoal_env python=3.9
conda activate gencoal_env
```

### Step 4: Install Dependencies

#### Install Conda Dependencies

Install all required packages listed in the `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will install all the necessary dependencies, including `numpy`, `pandas`, `rdkit`, and others.

#### Install Pip Dependencies

Some additional dependencies are managed via `pip`. Install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install any Python packages that were not covered by `conda`, such as specific libraries for machine learning or visualization.

### Step 5: Verify Installation

To ensure everything is set up correctly, you can run the following command to check if all dependencies are installed properly:

```bash
python -c "import pandas, numpy, torch, rdkit; print('All dependencies installed successfully!')"
```

If the command runs without errors, the environment is properly set up.

## File Descriptions

The following files are part of the `GenCoal` project's `/docs/examples` directories. These files serve as examples and documentation for generating coal molecular structures using different datasets and models.

- **bitumite.ipynb**: A Jupyter Notebook that contains the code and examples for generating molecular structures of bituminous coal. This file includes data processing, model inference, and visualization steps.
- **bitumite.json**: A JSON file containing configuration or data related to bituminous coal. This file is used to provide parameters or settings for the code in the `bitumite.ipynb` notebook.

- **clip_model_60000_parameters.pth**: A pre-trained PyTorch model file for the CLIP-based model. This model is used to infer or predict molecular structures from input data such as SMILES representations.

- **lignite.ipynb**: Similar to the `bitumite.ipynb`, this Jupyter Notebook demonstrates the generation of lignite coal molecular structures, providing step-by-step instructions on data processing and model usage.

- **lignite.json**: A JSON file containing configuration or data related to lignite coal. This file works with the `lignite.ipynb` notebook for setting model parameters or data inputs.

- **model_2023-04-14_TextEncoder_parameters.pkl**: A serialized Python pickle file containing the parameters of a text encoder model used in the molecular generation process. This model translates SMILES into molecular structure features.

- **smile_ir.csv**: A CSV file containing SMILES (Simplified Molecular Input Line Entry System) representations and corresponding infrared (IR) spectra data. This file is used as input data for training or testing models that predict molecular structures based on IR spectra.

- **text_features.npy**: A NumPy array file containing pre-processed text features, which are extracted from SMILES strings or other molecular representations. These features are used as input to machine learning models for structure prediction.

## Running GenCoal

To run the GenCoal software, follow the steps below. The main workflow is designed to be interactive, and users can execute each cell in the provided Jupyter Notebooks (`bitumite.ipynb` or `lignite.ipynb`) to generate coal molecular structures based on custom inputs.

### Step 1: Input Elemental Ratios

In the first part of the notebook, you will be prompted to input the elemental ratios of each element (except oxygen) on an air-dried basis (ADB), as well as the moisture and ash content. These values will serve as the basis for constructing the molecular model.

### Step 2: Add C13-NMR and IR Data

In the second part, you can upload your CNMR and IR spectra data in CSV format. The file should have two columns representing `x` and `y` values (e.g., chemical shift vs intensity for CNMR, and wavenumber vs absorbance for IR). This data will be used in the model construction process.

Make sure your CSV files are formatted as follows:

```
x, y
value1, value2
...
```

### Step 3: Input Desired Molecular Scale

In the third part, you will define the desired scale of the molecular model by specifying a list of carbon atom counts. You can modify the molecular scale either directly within the code block or by editing the associated JSON configuration file.

For example, in the code, you can modify the carbon atom count list like this:

```python
data["C_atom"] = [100, 200]
```

Alternatively, you can update the `C_atom` key in the corresponding JSON file.

### Output Results

After running the notebook and making the necessary adjustments, the output will display the predicted molecular structure of coal in SMILES format. You will also see several important results such as the chemical formula, the unsaturated carbon atom count, and the elemental ratio of the final coal model.

**Example Output**:

```
Predicted SMILES: CC1CCc2cnc3cc4cc5c(cc4cc3c2C1=O)Cc1c2c(c3c(C4CCC(C)C(=O)C4C4CCc6ncc7c8c(c9ccc%10c(c9c7c6C4=O)Cc4c(cc6ccc7cccc9ccc4c6c79)C%10)CCCC8C4C(=O)C(OCC6CCCCC6c6ccnc7ccc8c9c(ccc8c67)Cc6c(ccc7c6c6c(c8ccccc87)CC=C6)C9)CCC4Cc4c6ccccc6cc6c7c(ccc46)Cc4c(c6ccccc6c6ccccc46)C7)c4ccccc4cc3c1C5)C(C1c3ccc4ccc5cnc6c(c5c4c3Cc3cc4ccc5cccc7ccc(c31)c4c57)CC1CCCCC1C6=O)CCC2

Predicted Chemical Formula: C199H0O6N4S0

Unsaturated Carbon Atom Count: 147
Unsaturated Carbon Rate: 0.7387

Predicted Elemental Ratio: {'C': 94.02, 'H': 0.0, 'O': 3.78, 'N': 2.20, 'S': 0.0}
```

This output provides a comprehensive summary of the generated molecular structure, including key chemical properties and the predicted elemental composition of the coal model.

For further customization and analysis, you can modify the input data and parameters in the notebook and re-run the cells.

## Contributing

If you have a suggestion or find a bug, please post to our `Issues` page on GitHub.

## Questions

If you are having issues, please post to our `Issues` page on GitHub.

## Developers

- Haodong Liu ([liuhaodong.ncst@foxmail.com](mailto:liuhaodong.ncst@foxmail.com))

## Publications

https://doi.org/10.1016/j.energy.2024.130856
