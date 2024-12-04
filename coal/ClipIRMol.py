import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.Draw import MolsToGridImage
from scipy.interpolate import CubicSpline
import torch
import torch.nn.functional as F
import pickle
from coal import main
import random
from IPython.display import display
from rdkit.Chem.rdchem import BondType

# Globel variables
MODEL_FILENAME = 'clip_model_60000'
MODEL_PARAMS_FILENAME = MODEL_FILENAME + '_parameters.pth'
TEXT_ENCODER_PARAMS_PATH = 'model_2023-04-14_TextEncoder_parameters.pkl'
TEXT_FEATURES_PATH = "text_features.npy"


def check_isotope(molecule: Chem.Mol) -> bool:
    """
    Checks if all atoms in a molecule are isotopes with an isotope value of 0.

    This function iterates over all atoms in the provided molecule and checks if their isotope value 
    is 0. If all atoms have an isotope value of 0 (indicating no isotopes), the function returns 
    `True`. If any atom has a non-zero isotope value, indicating that it is an isotope, the function 
    returns `False`.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object to be checked. This object contains the atomic information of the molecule.

    Returns
    -------
    bool
        Returns `True` if no atoms in the molecule are isotopes (all atoms have isotope value 0).
        Returns `False` if at least one atom in the molecule is an isotope (has a non-zero isotope value).

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('CC(=O)C')
    >>> check_isotope(molecule)
    True

    >>> molecule_isotope = Chem.MolFromSmiles('CC(=O)[13C]')
    >>> check_isotope(molecule_isotope)
    False

    Notes
    -----
    - This function assumes that the molecule is represented as an RDKit `Chem.Mol` object.
    - An isotope is identified by checking if the `GetIsotope` method of the atom returns a value other than 0.
    """
    for atom in molecule.GetAtoms():
        if atom.GetIsotope() != 0:
            return False
    return True

def check_c_neigh_o(molecule: Chem.Mol) -> bool:
    """
    Checks if each carbon atom in the molecule has at most one oxygen neighbor.

    This function iterates over all carbon atoms in the provided molecule and checks the number 
    of oxygen atoms that are directly bonded to each carbon atom. If any carbon atom has more 
    than one oxygen neighbor, the function returns `False`. Otherwise, it returns `True`.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object to be checked. This object contains the atomic information of the molecule.

    Returns
    -------
    bool
        Returns `True` if all carbon atoms have at most one oxygen neighbor. 
        Returns `False` if any carbon atom has more than one oxygen neighbor.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('CC(=O)O')
    >>> check_c_neigh_o(molecule)
    True

    >>> molecule_invalid = Chem.MolFromSmiles('C(CO)O')
    >>> check_c_neigh_o(molecule_invalid)
    False

    Notes
    -----
    - This function assumes that the molecule is represented as an RDKit `Chem.Mol` object.
    - It checks only direct bonds between carbon and oxygen atoms.
    """
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'C':
            oxygen_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O']
            if len(oxygen_neighbors) > 1:
                return False
    return True

def check_rings(molecule: Chem.Mol) -> bool:
    """
    Checks the validity of rings in a molecule, ensuring that six-membered and five-membered rings 
    satisfy certain criteria defined by the helper functions `check_six_membered_ring` and `check_five_membered_ring`.

    The function examines all the rings present in the molecule. For each ring, if the ring contains six atoms, 
    it is passed to the `check_six_membered_ring` function for validation. Similarly, five-membered rings are 
    passed to the `check_five_membered_ring` function. Any ring that doesn't meet these criteria or is neither 
    five nor six members long will cause the function to return `False`. If all rings are valid, it returns `True`.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object whose rings are to be checked. This object contains the atomic and bonding information of the molecule.

    Returns
    -------
    bool
        Returns `True` if all rings in the molecule are either six-membered or five-membered and pass their respective checks. 
        Returns `False` if any ring is invalid (either it has a different number of atoms or fails the respective check).

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C1CCCCC1')
    >>> check_rings(molecule)
    True

    >>> molecule_invalid = Chem.MolFromSmiles('C1CCCC1')
    >>> check_rings(molecule_invalid)
    False

    Notes
    -----
    - This function assumes that the molecule is represented as an RDKit `Chem.Mol` object.
    - The helper functions `check_six_membered_ring` and `check_five_membered_ring` are used to validate specific ring types.
    """
    rings = molecule.GetRingInfo().AtomRings()  
    if not rings:
        return False

    for ring in rings:
        if len(ring) == 6:
            if not check_six_membered_ring(molecule, ring):
                return False
        elif len(ring) == 5:
            if not check_five_membered_ring(molecule, ring):
                return False
        else:
            return False
    
    return True

def check_six_membered_ring(molecule: Chem.Mol, ring: list) -> bool:
    """
    Validates whether a six-membered ring in a molecule satisfies specific structural criteria:
    - The ring should be a true six-membered ring.
    - The ring must contain exactly 6 double bonds.
    - The ring must contain no more than one nitrogen atom ('N').
    - If a nitrogen atom is present, it should not have a total degree of 3 (i.e., it should not be a tertiary nitrogen).

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.
    
    ring : list
        A list of atom indices representing the six-membered ring in the molecule. The atoms in the ring are indexed by their position in the molecule.

    Returns
    -------
    bool
        Returns `True` if the six-membered ring satisfies all the structural criteria, otherwise `False`.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C1CCCCC1')
    >>> ring = molecule.GetRingInfo().AtomRings()[0]
    >>> check_six_membered_ring(molecule, ring)
    True

    >>> molecule_invalid = Chem.MolFromSmiles('C1CCNCC1')
    >>> ring = molecule_invalid.GetRingInfo().AtomRings()[0]
    >>> check_six_membered_ring(molecule_invalid, ring)
    False

    Notes
    -----
    - The function checks if the ring contains exactly 6 double bonds (`double_bonds == 6`) and no more than one nitrogen atom.
    - If nitrogen is present in the ring, it ensures that it is not a tertiary nitrogen (degree == 3).
    - The function also validates that the atom at index 0 of the ring is in a six-membered ring.
    """
    # Get the atoms in the ring using their indices
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    
    # Count the number of double bonds in the ring (SP2 hybridization)
    double_bonds = sum(1 for atom in ring_atoms if atom.GetHybridization() == rdchem.HybridizationType.SP2)
    
    # Count the number of nitrogen atoms in the ring
    n_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'N')
    
    # Check if the ring is a true six-membered ring, contains exactly 6 double bonds, and no more than 1 nitrogen
    if not molecule.GetRingInfo().IsAtomInRingOfSize(ring_atoms[0].GetIdx(), 6) or double_bonds != 6 or n_count > 1:
        return False

    # Check if any nitrogen atom has a degree of 3 (tertiary nitrogen) in the ring
    for atom in ring_atoms:
        if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3:
            return False

    return True

def check_five_membered_ring(molecule: Chem.Mol, ring: list) -> bool:
    """
    Validates whether a five-membered ring in a molecule satisfies specific structural criteria:
    - The ring should contain exactly 4 double bonds.
    - The ring should contain no more than one heteroatom, which can be either nitrogen (N) or sulfur (S).

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.
    
    ring : list
        A list of atom indices representing the five-membered ring in the molecule. The atoms in the ring are indexed by their position in the molecule.

    Returns
    -------
    bool
        Returns `True` if the five-membered ring satisfies all the structural criteria, otherwise `False`.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C1CCC2C2')
    >>> ring = molecule.GetRingInfo().AtomRings()[0]
    >>> check_five_membered_ring(molecule, ring)
    True

    >>> molecule_invalid = Chem.MolFromSmiles('C1CCN2C2')
    >>> ring = molecule_invalid.GetRingInfo().AtomRings()[0]
    >>> check_five_membered_ring(molecule_invalid, ring)
    False

    Notes
    -----
    - The function checks if the ring contains exactly 4 double bonds (`double_bonds == 4`) and no more than one heteroatom (N or S).
    """
    # Get the atoms in the ring using their indices
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    
    # Count the number of double bonds in the ring (SP2 hybridization)
    double_bonds = sum(1 for atom in ring_atoms if atom.GetHybridization() == rdchem.HybridizationType.SP2)
    
    # Count the number of heteroatoms (N or S) in the ring
    ns_count = sum(1 for atom in ring_atoms if atom.GetSymbol() in ['N', 'S'])
    
    # Check if the ring contains exactly 4 double bonds and no more than 1 heteroatom (N or S)
    return double_bonds == 4 and ns_count <= 1

def check_ring_no_oxy_sulph(molecule: Chem.Mol) -> bool:
    """
    Checks if a molecule's rings contain any oxygen atoms or more than one heteroatom (nitrogen or sulfur).
    The function returns `False` if any of the following conditions are met in any ring:
    - The ring contains an oxygen atom.
    - The ring contains more than one heteroatom (nitrogen or sulfur).
    
    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.
    
    Returns
    -------
    bool
        Returns `True` if none of the rings in the molecule contain oxygen atoms or more than one heteroatom.
        Returns `False` if any ring contains an oxygen atom or more than one heteroatom.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C1CC2CCO2C1')
    >>> check_ring_no_oxy_sulph(molecule)
    False

    >>> molecule_no_oxy_sulph = Chem.MolFromSmiles('C1CCCC2C2')
    >>> check_ring_no_oxy_sulph(molecule_no_oxy_sulph)
    True

    Notes
    -----
    - The function checks each ring in the molecule for the presence of oxygen (`O`) and heteroatoms (`N`, `S`).
    - It returns `False` if any ring contains oxygen or more than one heteroatom.
    """
    # Retrieve information on all rings in the molecule
    rings = molecule.GetRingInfo().AtomRings()  
    
    for ring in rings:
        # Get atoms in the current ring using their indices
        ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
        
        # Count the number of heteroatoms (N or S) in the ring
        oxy_sulph_count = sum(atom.GetSymbol() in ['N', 'S'] for atom in ring_atoms)
        
        # Check for oxygen atoms in the ring
        for idx in ring:
            atom = molecule.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'O':
                return False
        
        # If there are more than one heteroatoms (N or S), return False
        if oxy_sulph_count > 1:
            return False
    
    # Return True if no rings contain oxygen or more than one heteroatom
    return True

def check_invalid_bonds(molecule: Chem.Mol) -> bool:
    """
    Checks if a molecule contains any invalid bonds. Invalid bonds are defined as:
    - Nitrogen (N) bonded to Oxygen (O)
    - Oxygen (O) bonded to Oxygen (O)
    - Sulfur (S) bonded to Oxygen (O)
    - Sulfur (S) bonded to Sulfur (S)
    
    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.
    
    Returns
    -------
    bool
        Returns `True` if no invalid bonds are found in the molecule, and `False` if any invalid bonds are detected.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C1CCO1')  # valid bond
    >>> check_invalid_bonds(molecule)
    True

    >>> molecule_invalid = Chem.MolFromSmiles('C1SNO1')  # invalid bond (S-N)
    >>> check_invalid_bonds(molecule_invalid)
    False

    Notes
    -----
    - The function checks all bonds in the molecule to see if any of them match the invalid combinations.
    - If any invalid bond is found, the function immediately returns `False`.
    """
    # Iterate through each bond in the molecule
    for bond in molecule.GetBonds():
        # Get the atom symbols for the two atoms involved in the bond
        bond_atoms = set([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()])
        
        # Define a set of invalid bond combinations
        invalid_combinations = [{'N', 'O'}, {'O', 'O'}, {'S', 'O'}, {'S', 'S'}]
        
        # Check if the current bond is one of the invalid combinations
        if bond_atoms in invalid_combinations:
            return False
    
    # Return True if no invalid bonds are found
    return True

def check_chain_double_bonds(molecule: Chem.Mol) -> bool:
    """
    Checks if the molecule has any non-ring chain double bonds (i.e., double bonds between carbon atoms that 
    are not part of a ring structure). If such bonds are found, the function returns `False`; otherwise, it returns `True`.

    This function checks each carbon-carbon bond in the molecule to determine if:
    - The bond is a double bond.
    - Both carbon atoms involved in the double bond are not part of any ring structure.
    
    If both conditions are met, the function returns `False`, indicating that the molecule contains an invalid non-ring 
    double bond. If no such bonds are found, the function returns `True`.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.

    Returns
    -------
    bool
        Returns `True` if there are no non-ring carbon-carbon double bonds in the molecule.
        Returns `False` if there is a non-ring carbon-carbon double bond.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('C=C')  # a simple carbon-carbon double bond
    >>> check_chain_double_bonds(molecule)
    True  # because both carbon atoms are not part of a ring

    >>> molecule_invalid = Chem.MolFromSmiles('C1CC=CC1')  # a ring with a carbon-carbon double bond
    >>> check_chain_double_bonds(molecule_invalid)
    True  # valid ring with a double bond
    
    >>> molecule_invalid = Chem.MolFromSmiles('C=C=C')  # a chain with a non-ring carbon-carbon double bond
    >>> check_chain_double_bonds(molecule_invalid)
    False  # non-ring carbon-carbon double bond found

    Notes
    -----
    - The function uses RDKit's `GetRingInfo().IsAtomInRingOfSize()` method to check if atoms are part of a ring structure.
    - The function only checks single and double bonds between carbon atoms, ignoring other bonds.
    """
    # Iterate through each bond in the molecule
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

        # Check if the bond is a carbon-carbon bond
        if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
            # Check if each carbon atom is part of any ring (size 3 or greater)
            atom1_in_ring = any(molecule.GetRingInfo().IsAtomInRingOfSize(atom1.GetIdx(), ring_size) for ring_size in range(3, molecule.GetNumAtoms() + 1))
            atom2_in_ring = any(molecule.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), ring_size) for ring_size in range(3, molecule.GetNumAtoms() + 1))

            # If the bond is not a single bond and the atoms are not part of any ring, return False
            if not (atom1_in_ring and atom2_in_ring) and bond.GetBondType() != rdchem.BondType.SINGLE:
                return False
    
    # Return True if no invalid non-ring double bonds are found
    return True

def check_chain_n_c(molecule: Chem.Mol) -> bool:
    """
    Checks if all nitrogen atoms (N) in the molecule are connected to carbon (C) atoms that are part of a six-membered ring.
    If any nitrogen-carbon bond is not part of a six-membered ring, the function returns `False`; otherwise, it returns `True`.

    This function iterates through each nitrogen atom in the molecule and checks whether each nitrogen is connected to 
    a carbon atom, and whether both atoms involved in the bond are part of a six-membered ring.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        The RDKit molecule object containing the atomic and bonding information of the molecule.

    Returns
    -------
    bool
        Returns `True` if all nitrogen atoms are connected to carbon atoms that are part of a six-membered ring.
        Returns `False` if any nitrogen-carbon bond is not part of a six-membered ring.

    Example
    -------
    >>> molecule = Chem.MolFromSmiles('NCCN')  # a chain with nitrogen atoms connected to carbon atoms
    >>> check_chain_n_c(molecule)
    False  # because the N-C bonds are not part of six-membered rings
    
    >>> molecule_valid = Chem.MolFromSmiles('C1CCCCNC1')  # a six-membered ring with nitrogen and carbon atoms
    >>> check_chain_n_c(molecule_valid)
    True  # because the N-C bonds are part of a six-membered ring
    
    Notes
    -----
    - The function uses RDKit's `GetRingInfo().IsAtomInRingOfSize()` method to check if atoms are part of a six-membered ring.
    - Only checks nitrogen-carbon bonds, ignoring other types of bonds.
    """
    # Iterate through each atom in the molecule
    for atom in molecule.GetAtoms():
        # Check if the atom is nitrogen
        if atom.GetSymbol() == 'N':
            # Check all neighbors of the nitrogen atom
            for neighbor in atom.GetNeighbors():
                # Check if the neighbor is carbon
                if neighbor.GetSymbol() == 'C':
                    # Ensure both the nitrogen and carbon are in a six-membered ring
                    if not (molecule.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), 6) and molecule.GetRingInfo().IsAtomInRingOfSize(neighbor.GetIdx(), 6)):
                        return False
    
    # Return True if all nitrogen-carbon bonds are part of six-membered rings
    return True

def process_image(image_path: str) -> np.ndarray:
    """
    Processes a CSV file containing X and Y data, applies cubic spline interpolation, 
    and reshapes the resulting data for further use.

    This function reads a CSV file where the X and Y values are stored in two columns labeled 'X' and 'Y'. 
    It ensures that the X values are sorted in ascending order and performs cubic spline interpolation 
    to generate a smoother curve for the range from 400 to 4000 with a step size of 2. 
    The interpolated data is then reshaped into a 4D numpy array with shape (2, 1, 36, 50).

    Parameters
    ----------
    image_path : str
        The file path to the CSV file containing the X and Y data. 
        The CSV is expected to have two columns labeled 'X' and 'Y', representing coordinates.

    Returns
    -------
    np.ndarray
        A 4D numpy array with shape (2, 1, 36, 50), where the first dimension corresponds to the number 
        of channels (2), the second dimension corresponds to a single set of data (1), 
        the third dimension corresponds to the interpolated X values (36), and the fourth dimension 
        corresponds to the number of interpolated Y values (50).

    Notes
    -----
    - The function checks if the X values are sorted in ascending order and sorts them if necessary.
    - Cubic spline interpolation is applied to generate smoother Y values over a specific range of X values.
    - The reshaped output is prepared in a specific format, which may be useful for feeding into neural networks or other processing pipelines.
    
    Example
    -------
    >>> process_image("data.csv")
    array([[[[0.1, 0.2, 0.3, ..., 0.4, 0.5],
             [0.6, 0.7, 0.8, ..., 0.9, 1.0],
             ...]]])
    """
    
    # Read the CSV file
    df = pd.read_csv(image_path)
    
    # Ensure X values are sorted
    if not all(df['X'].diff()[1:] > 0):
        df = df.sort_values(by='X', ascending=True)

    # Extract X and Y data
    x_data = df['X'].values
    y_data = df['Y'].values

    # Perform cubic spline interpolation
    cs = CubicSpline(x_data, y_data)
    x = np.arange(400, 4000, 2)
    y = cs(x)

    # Create a DataFrame with interpolated data
    data = {'X': x, 'Y': y}
    selected_df = pd.DataFrame(data)

    # Convert to numpy array and reshape
    selected_data = selected_df.astype('float32').to_numpy()
    return selected_data.reshape(2, 1, 36, 50)

def load_model_and_features() -> tuple:
    """
    Loads the pre-trained CLIP model and associated text features from specified paths.

    This function performs the following tasks:
    1. Loads text encoder parameters from a binary file.
    2. Initializes the text encoder model using the loaded parameters.
    3. Loads the pre-trained CLIP model's parameters from a specified file.
    4. Initializes the CLIP model with the loaded text encoder.
    5. Loads pre-computed text features from a specified file.

    The function returns a tuple containing:
    - The initialized CLIP model.
    - The loaded text features.

    Returns
    -------
    tuple
        A tuple where:
        - The first element is the CLIP model (`main.CLIP`), initialized with the loaded parameters.
        - The second element is a numpy array containing the loaded text features.

    Notes
    -----
    - The model parameters are loaded using `torch.load` and the text encoder parameters are loaded using `pickle`.
    - The file paths for the text encoder parameters, model parameters, and text features are expected to be predefined in the global constants.
    
    Example
    -------
    >>> loaded_clip_model, loaded_text_features = load_model_and_features()
    >>> print(loaded_clip_model)
    <CLIP model object>
    >>> print(loaded_text_features.shape)
    (1000, 512)
    """

    # Load text encoder parameters from the pickle file
    with open(TEXT_ENCODER_PARAMS_PATH, 'rb') as f:
        text_encoder_params = pickle.load(f)

    # Initialize the text encoder model with the loaded parameters
    text_encoder = main.TextEncoder(text_encoder_params)

    # Load the pre-trained model parameters (CLIP model)
    loaded_model_params = torch.load(MODEL_PARAMS_FILENAME)

    # Initialize the CLIP model with the text encoder
    loaded_clip_model = main.CLIP(text_encoder)

    # Load the model parameters into the CLIP model
    loaded_clip_model.load_state_dict(loaded_model_params)

    # Load pre-computed text features from a numpy file
    loaded_text_features = np.load(TEXT_FEATURES_PATH)

    # Return the initialized CLIP model and text features
    return loaded_clip_model, loaded_text_features

def compute_similarity(image_input: np.ndarray, clip_model: torch.nn.Module, text_features: np.ndarray) -> np.ndarray:
    """
    Computes the similarity between image features and precomputed text features using the CLIP model.

    This function performs the following tasks:
    1. Moves the model and inputs to the appropriate device (CPU or GPU).
    2. Encodes the input image into feature embeddings using the CLIP model.
    3. Normalizes both the image and text features.
    4. Computes the cosine similarity between the normalized image features and the normalized text features.
    
    The similarity score is computed as the dot product of the normalized image and text feature vectors,
    scaled by a factor of 100.

    Parameters
    ----------
    image_input : np.ndarray
        The input image data as a numpy array. The shape of the input should be compatible with the CLIP model.
        Typically, this should be of shape (1, C, H, W), where C is the number of channels, H is the height, 
        and W is the width of the image.

    clip_model : torch.nn.Module
        The pre-trained CLIP model, used to generate image features from the input image.

    text_features : np.ndarray
        A numpy array of precomputed text features. Each row in this array represents a feature vector
        corresponding to a text input.

    Returns
    -------
    np.ndarray
        A numpy array containing the similarity scores between the image features and each of the precomputed text features.
        The similarity score is scaled by a factor of 100 and represents the cosine similarity between the vectors.

    Notes
    -----
    - The function assumes the input image and text features are already preprocessed and ready for similarity computation.
    - The function moves all tensors to the appropriate device (CPU or GPU) based on availability of CUDA.
    - The similarity scores are calculated by first normalizing the image and text feature vectors, and then computing their dot product.

    Example
    -------
    >>> image_input = np.random.randn(1, 3, 224, 224)  # Example image data
    >>> clip_model = load_clip_model()  # Load your pre-trained CLIP model
    >>> text_features = np.random.randn(10, 512)  # Example text features
    >>> similarity_scores = compute_similarity(image_input, clip_model, text_features)
    >>> print(similarity_scores.shape)
    (1, 10)  # Similarity score between the image and each of the 10 text features
    """
    
    # Determine the device (CPU or GPU) and move the model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device).eval()

    # Convert the image input to a tensor and move it to the device
    image_input = torch.from_numpy(image_input).float().to(device)

    # Compute the image features using the CLIP model
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()

    # Extract and move image features to CPU
    image_features = image_features[0].unsqueeze(0).cpu().numpy()

    # Normalize the image and text features
    normalized_image_features = F.normalize(torch.from_numpy(image_features), dim=1).numpy()
    normalized_text_features = F.normalize(torch.from_numpy(text_features), dim=1).numpy()

    # Compute the cosine similarity scores
    similarity_scores = 100.0 * np.dot(normalized_image_features, normalized_text_features.T)

    return similarity_scores

def skip_and_update_top_n_ranking(similarity_scores: np.ndarray, top_n: int, df: pd.DataFrame) -> list:
    """
    Sorts molecules by similarity scores and selects the top N molecules that satisfy certain conditions.
    
    This function evaluates each molecule based on its similarity scores and checks whether it satisfies 
    a set of predefined conditions. If a molecule satisfies all conditions, it is considered valid, 
    and the top N valid molecules are selected based on the highest similarity scores.
    
    Parameters
    ----------
    similarity_scores : np.ndarray
        A 2D numpy array where each row corresponds to a molecule, 
        and each column corresponds to the similarity score between the molecule and other molecules. 
        Shape: (num_molecules, num_comparisons).
        
    top_n : int
        The number of top valid molecules to select based on the highest similarity scores.
        
    df : pd.DataFrame
        A pandas DataFrame containing the SMILES strings of the molecules. 
        The DataFrame should have a column labeled 'smiles' that contains the SMILES representations.
    
    Returns
    -------
    updated_predicted_smiles_and_scores : list
        A list where each element corresponds to a molecule (identified by its index) and contains the 
        top N valid molecules (as SMILES strings) along with their similarity scores. 
        Each inner list contains tuples of (SMILES, score) for the top N molecules.
        
    Notes
    -----
    - The molecules are selected based on their similarity scores, with the highest scores being considered first.
    - The function filters out molecules that do not satisfy all the conditions defined below.
    - If there are fewer than `top_n` valid molecules, only the valid molecules will be included in the result.
    """
    
    updated_predicted_smiles_and_scores = []
    
    for i in range(similarity_scores.shape[0]):
        valid_smiles_list = []
        
        sorted_indices = np.argsort(-similarity_scores[i])

        for index in sorted_indices:
            smiles = list(df['smiles'])[index]
            molecule = Chem.MolFromSmiles(smiles)
            
            # Check all conditions
            if (
                check_isotope(molecule) and
                check_c_neigh_o(molecule) and
                check_rings(molecule) and
                check_ring_no_oxy_sulph(molecule) and
                check_invalid_bonds(molecule) and
                check_chain_double_bonds(molecule) and
                check_chain_n_c(molecule)
            ):
                score = similarity_scores[i, index]
                valid_smiles_list.append((smiles, score))
                
                if len(valid_smiles_list) == top_n:
                    break
        
        updated_predicted_smiles_and_scores.append(valid_smiles_list)
    
    return updated_predicted_smiles_and_scores

def retrieve_small_molecules(image_path: str) -> list:
    """
    Retrieves the top N small molecules whose SMILES strings are most similar to the input image using a CLIP model.

    This function performs the following steps:
    1. Reads the SMILES data from a CSV file ('smile_ir.csv').
    2. Processes the input image (specified by `image_path`) into a format suitable for the CLIP model.
    3. Loads the pre-trained CLIP model and precomputed text features.
    4. Computes the similarity scores between the image features and the text features using the CLIP model.
    5. Sorts the molecules by similarity score and selects the top N molecules.
    6. Returns the SMILES strings of the top N molecules along with their corresponding indices and similarity scores.

    Parameters
    ----------
    image_path : str
        The file path of the input image to be compared to the SMILES features. This image will be processed and used to compute similarity scores.

    Returns
    -------
    list
        A list of dictionaries, each containing:
        - 'smiles': The SMILES string of the molecule.
        - 'index': The index of the molecule in the original SMILES dataset.
        - 'score': The similarity score between the image and the molecule's SMILES.

    Notes
    -----
    - The function assumes that a CSV file named 'smile_ir.csv' exists, which contains the SMILES strings in a column labeled 'smiles'.
    - The `process_image` function is used to convert the input image into a format that the CLIP model can process.
    - The `skip_and_update_top_n_ranking` function is used to rank the molecules by similarity score and select the top N.
    - The pre-trained CLIP model is used to compute the similarity scores based on the image and the precomputed text features.

    Example
    -------
    >>> image_path = 'path/to/image.png'  # Example image file path
    >>> results = retrieve_small_molecules(image_path)
    >>> for result in results:
    >>>     print(result)
    {'smiles': 'CCO', 'index': 1, 'score': 95.6}
    {'smiles': 'CCN', 'index': 2, 'score': 93.4}
    """

    # Read the SMILES data from the CSV file
    df = pd.read_csv('smile_ir.csv')

    # Process the input image
    image_input = process_image(image_path)

    # Load the pre-trained CLIP model and text features
    loaded_clip_model, loaded_text_features = load_model_and_features()

    # Compute similarity scores between the image input and the text features
    similarity_scores = compute_similarity(image_input, loaded_clip_model, loaded_text_features)

    # Set the top N molecules to retrieve
    top_n = 40
    updated_predicted_smiles_and_scores = skip_and_update_top_n_ranking(similarity_scores, top_n, df)

    # Prepare the results as a list of dictionaries
    results = []
    for predicted_smiles_list in updated_predicted_smiles_and_scores:
        for smiles, score in predicted_smiles_list:
            index = list(df['smiles']).index(smiles)
            results.append({
                'smiles': smiles,
                'index': index,
                'score': score
            })

    return results

def convert_data_to_smiles_scores(retrieved_molecules: list) -> list:
    """
    Converts the retrieved molecules data into a list of (SMILES, score) tuples.

    This function takes the data of retrieved molecules, where each entry contains a 'smiles' and 'score' key,
    and converts it into a list of tuples. Each tuple contains the SMILES string and the corresponding similarity score.

    Parameters
    ----------
    retrieved_molecules : list
        A list of dictionaries, where each dictionary contains the following keys:
        - 'smiles': The SMILES string representing the molecule.
        - 'score': The similarity score of the molecule.

    Returns
    -------
    list
        A list of tuples, where each tuple contains:
        - A SMILES string representing a molecule.
        - A similarity score of the molecule.
    
    Example
    -------
    >>> retrieved_molecules = [
    >>>     {'smiles': 'CCO', 'score': 95.6},
    >>>     {'smiles': 'CCN', 'score': 93.4}
    >>> ]
    >>> convert_data_to_smiles_scores(retrieved_molecules)
    [('CCO', 95.6), ('CCN', 93.4)]
    """
    return [(entry['smiles'], entry['score']) for entry in retrieved_molecules]

def display_molecules(smiles_and_scores: list, mols_per_row: int = 5, img_size: tuple = (300, 300)) -> None:
    """
    Create and display a grid image of molecules with their associated scores.

    This function takes a list of (SMILES, score) tuples, converts the SMILES strings into RDKit molecule objects,
    and generates a grid image of the molecules. Each molecule will be labeled with its corresponding SMILES string 
    and similarity score. The image will be displayed in a grid format.

    Parameters
    ----------
    smiles_and_scores : list
        A list of tuples, where each tuple contains:
        - A SMILES string representing the molecule.
        - A similarity score associated with the molecule.
    
    mols_per_row : int, optional, default=5
        The number of molecules to display per row in the grid. The default value is 5.

    img_size : tuple, optional, default=(300, 300)
        The size of each individual image (width, height) in the grid. The default value is (300, 300).

    Returns
    -------
    None
        This function does not return anything. It displays the image in the output.

    Example
    -------
    >>> smiles_and_scores = [
    >>>     ('CCO', 95.6),
    >>>     ('CCN', 93.4),
    >>>     ('CCCl', 89.2)
    >>> ]
    >>> display_molecules(smiles_and_scores, mols_per_row=3, img_size=(200, 200))
    """
    
    # Convert SMILES strings to RDKit molecule objects
    predicted_molecules = [Chem.MolFromSmiles(smiles) for smiles, _ in smiles_and_scores]
    
    # Create legends for each molecule
    legends = [f"{i+1}. {smiles} (score: {score:.4f})" for i, (smiles, score) in enumerate(smiles_and_scores)]
    
    # Create the grid image
    img = MolsToGridImage(predicted_molecules, molsPerRow=mols_per_row, subImgSize=img_size, legends=legends)
    
    display(img)
    return

def show_atom_number(mol: Chem.Mol, label: str) -> Chem.Mol:
    """
    Annotates each atom in the molecule with its atom index.

    This function iterates through all the atoms in a given RDKit molecule and sets each atom's property
    with the provided label as its atom index. This can be useful for visualizing or labeling atoms in a 
    molecule, especially when working with graphical representations of molecular structures.

    Parameters
    ----------
    mol : Chem.Mol
        An RDKit molecule object to which atom indices will be added.
    
    label : str
        The label (or property name) to associate with each atom. The atom index will be stored under this label.

    Returns
    -------
    Chem.Mol
        The input molecule with atom indices set as properties.

    Example
    -------
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> labeled_mol = show_atom_number(mol, 'atom_index')
    >>> for atom in labeled_mol.GetAtoms():
    >>>     print(atom.GetProp('atom_index'))
    """
    
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    
    return mol

def find_required_carbons(smiles: str) -> list[int]:
    """
    Finds carbon atoms in a molecule that are connected to exactly two carbon atoms and one hydrogen atom.

    This function processes a given SMILES string to identify specific carbon atoms in the structure 
    that meet the following conditions:
    - The carbon atom must be connected to exactly two other carbon atoms.
    - The carbon atom must be connected to exactly one hydrogen atom.

    Parameters
    ----------
    smiles : str
        The SMILES representation of the molecule to analyze.

    Returns
    -------
    list[int]
        A list of indices of the carbon atoms that satisfy the conditions described above.

    Example
    -------
    >>> find_required_carbons('CCO')
    [1]
    >>> find_required_carbons('C1CCCCC1')
    [1, 3, 4, 6]
    """
    
    # 为SMILES补充H原子
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
    # 创建一个列表来保存满足条件的C原子的索引
    required_carbons = []

    # 遍历分子中的所有原子
    for atom in molecule.GetAtoms():
        # 判断该原子是否为C原子
        if atom.GetSymbol() == "C":
            # 获取该原子的所有邻居原子
            neighbors = [neighbor for neighbor in atom.GetNeighbors()]
            
            # 计算该原子连接的C原子数量
            carbon_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() == 'C' or neighbor.GetSymbol() == 'N'])

            # 计算该原子连接的H原子数量
            hydrogen_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() == 'H'])

            # 如果该C原子只与2个C原子相连，并且有1个H原子
            if carbon_neighbors == 2 and hydrogen_neighbors == 1:
                required_carbons.append(atom.GetIdx())
    
    return required_carbons

def connect_rings_C4(smiles1: str) -> str:
    """
    Connects a molecule, represented by a SMILES string `smiles1`, with a butane molecule ('CCCC'),
    using two suitable carbon atoms from `smiles1` that meet specific connection conditions.
    
    This function performs the following steps:
    1. Finds suitable carbon atoms in `smiles1` that are connected to two other carbon atoms and one hydrogen atom.
    2. Randomly selects one pair of adjacent suitable carbon atoms.
    3. Connects the selected pair of carbon atoms from `smiles1` to two carbon atoms in the butane molecule (`smiles2 = 'CCCC'`).
    4. The bond type between the connected atoms is set to aromatic.
    5. Returns the SMILES representation of the new molecule with the combined structure.

    Parameters
    ----------
    smiles1 : str
        The SMILES string representing the first molecule (which will be connected to butane).
    
    Returns
    -------
    str
        The SMILES representation of the new molecule after connecting the rings.
        If no suitable carbon atoms are found in `smiles1`, the function returns `None`.
    
    Example
    -------
    >>> connect_rings_C4('CC1=CC2C=CC(C)=C2C=C1')
    'CC1=CC2C=CC(C)=C2C=C1CCCC'
    """
    
    smiles2 = 'CCCC'  # Butane molecule
    carbons2 = [0, 3]  # Indices of carbon atoms in butane that will be connected
    
    # Find the required carbon atoms in the first molecule
    carbons1 = find_required_carbons(smiles1)

    # Check if no suitable carbon atom was found in the molecule
    if not carbons1:
        print("No suitable carbon atom found in the molecule")
        return None

    # Convert the SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Find all pairs of adjacent suitable carbon atoms in mol1
    index_pairs_carbons1 = []
    
    for i in range(len(carbons1)):
        atom1 = mol1.GetAtomWithIdx(carbons1[i])
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() in carbons1:
                if carbons1[i] < neighbor.GetIdx():  # Avoid duplicates
                    index_pairs_carbons1.append((carbons1[i], neighbor.GetIdx()))

    # Randomly choose one pair of carbon atoms to connect
    chosen_pair1 = random.choice(index_pairs_carbons1)

    # Combine the two molecules (mol1 and mol2)
    combined = Chem.CombineMols(mol1, mol2)
    
    # Create an editable molecule from the combined molecule
    edit_combined = Chem.EditableMol(combined)

    # Recalculate atom indices after combination
    index1 = chosen_pair1[0]  # carbon atom from mol1
    index2 = chosen_pair1[1]  # carbon atom from mol1
    index3 = len(mol1.GetAtoms())  # first atom of mol2 (butane)
    index4 = len(mol1.GetAtoms()) + 3  # last atom of mol2 (butane)

    # Define a set to store the indices of all atoms in butane in the merged molecule
    butane_indices = set(range(len(mol1.GetAtoms()), len(mol1.GetAtoms()) + 4))
    butane_indices.add(index1)
    butane_indices.add(index2)

    # Add single bonds between the selected carbons in mol1 and the carbons in mol2
    edit_combined.AddBond(index1, index3, order=BondType.SINGLE)
    edit_combined.AddBond(index2, index4, order=BondType.SINGLE)

    # Get the modified molecule
    connected_mol = edit_combined.GetMol()

    # Change the bond type of the bonds within butane to AROMATIC
    for idx in butane_indices:
        atom = connected_mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in butane_indices:
                bond = connected_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond is not None:
                    bond.SetBondType(rdchem.BondType.AROMATIC)

    # Sanitize the molecule to ensure it is chemically valid
    Chem.SanitizeMol(connected_mol)

    # Return the SMILES representation of the connected molecule
    return Chem.MolToSmiles(connected_mol)

def find_required_aldehyde_carbons(smiles: str) -> list:
    """
    Find and return the indices of carbon atoms in the given molecule that are part of an aldehyde group.
    Aldehyde groups have the structure -CHO, where the carbon is bonded to one hydrogen, one oxygen, 
    and one other carbon atom.

    Parameters
    ----------
    smiles : str
        The SMILES string representation of the molecule.

    Returns
    -------
    list
        A list of indices of carbon atoms that are part of an aldehyde group. If no such carbon atoms 
        are found, an empty list is returned.

    Example
    -------
    >>> find_required_aldehyde_carbons('CC(C)C=O')
    [3]
    >>> find_required_aldehyde_carbons('CCO')
    []
    """
    # Convert the SMILES string to a molecule object and add hydrogen atoms
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # List to store the indices of carbon atoms that are part of an aldehyde group
    carbons = []

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':  # Only check carbon atoms
            neighbors = atom.GetNeighbors()

            # Check if the carbon atom is bonded to exactly 3 atoms (C, H, and O)
            if len(neighbors) == 3:
                symbols = [neighbor.GetSymbol() for neighbor in neighbors]
                # If the carbon has one C, one H, and one O as neighbors, it is part of an aldehyde
                if symbols.count('C') == 1 and symbols.count('H') == 1 and symbols.count('O') == 1:
                    carbons.append(atom.GetIdx())  # Add the carbon atom index to the list

    return carbons

def satisfy_beta_carbons_conditions(smiles: str) -> bool:
    """
    Check if the molecule's alpha and beta carbons satisfy certain conditions around the aldehyde group:
    - The beta carbons, which are connected to alpha carbons, must have at least one neighbor
      with fewer than 3 neighbors (i.e., connected to a hydrogen or a simple carbon-carbon bond).
    
    Parameters
    ----------
    smiles : str
        The SMILES string representation of the molecule.

    Returns
    -------
    bool
        Returns `True` if all beta carbons connected to alpha carbons satisfy the condition; 
        otherwise, returns `False`.
    
    Example
    -------
    >>> satisfy_beta_carbons_conditions('CC(C)C=O')
    True
    >>> satisfy_beta_carbons_conditions('CCO')
    False
    """
    # Convert the SMILES string to a molecule object and add hydrogen atoms
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # Get the list of aldehyde carbon indices (those involved in the aldehyde group)
    aldehyde_carbons = find_required_aldehyde_carbons(smiles)

    # Iterate over each aldehyde carbon
    for aldehyde_carbon in aldehyde_carbons:
        # Get the alpha carbons (neighbors of the aldehyde carbon that are carbons)
        alpha_carbons = [neighbor for neighbor in mol.GetAtomWithIdx(aldehyde_carbon).GetNeighbors() if neighbor.GetSymbol() == 'C']

        # If there are no alpha carbons connected to this aldehyde carbon, return False
        if not alpha_carbons:
            return False

        # Iterate over each alpha carbon
        for alpha_carbon in alpha_carbons:
            # Get the beta carbons (neighbors of the alpha carbon that are carbons)
            beta_carbons = [neighbor for neighbor in alpha_carbon.GetNeighbors() if neighbor.GetSymbol() == 'C']
            # Remove the aldehyde carbon from the beta carbon list
            beta_carbons = [beta for beta in beta_carbons if beta.GetIdx() != aldehyde_carbon]

            # Check if any of the beta carbons has a neighbor with fewer than 3 neighbors (which should be a hydrogen)
            if not any(['H' in [n.GetSymbol() for n in beta.GetNeighbors()] for beta in beta_carbons]):
                return False

    return True

def filter_smiles_and_scores(updated_predicted_smiles_and_scores):
    """
    Filters the given list of SMILES strings and their associated similarity scores based on 
    whether they satisfy the beta carbon conditions.

    Parameters
    ----------
    updated_predicted_smiles_and_scores : list of tuple
        A list of tuples, where each tuple contains a SMILES string and its corresponding score.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains a SMILES string and its corresponding score 
        that satisfy the beta carbon conditions.

    Example
    -------
    >>> updated_predicted_smiles_and_scores = [('CCO', 0.95), ('CC(C)C=O', 0.92)]
    >>> filter_smiles_and_scores(updated_predicted_smiles_and_scores)
    [('CC(C)C=O', 0.92)]
    """
    # Store the filtered SMILES and scores
    filtered_smiles_and_scores = []
    
    # Iterate over the predicted SMILES and scores
    for smiles_and_score in updated_predicted_smiles_and_scores[0]:
        # Extract SMILES and score
        smiles, score = smiles_and_score
        
        # Check if the SMILES satisfies the beta carbon conditions
        if satisfy_beta_carbons_conditions(smiles):
            # Add to the filtered list if it satisfies the condition
            filtered_smiles_and_scores.append((smiles, score))
    
    return filtered_smiles_and_scores
