from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import AddHs
from rdkit.Chem import rdchem
from rdkit.Chem import Draw
from rdkit.Chem import Draw, AllChem
from IPython.display import display

from collections import defaultdict
import copy
import math
import ast
import random
import json

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import copy

# NMR integral ------------ create json
def calculate_C90_C180(file_path):
    # 导入数据
    df = pd.read_csv(file_path)
    # 数据清洗
    filtered_df = df[(df['X'] >= 0) & (df['X'] <= 250) & (df['Y'] >= 0)]
    
    # 取出该列的所有数据
    x_list = filtered_df['X'].values
    y_list = filtered_df['Y'].values
    
    x = np.array(x_list)
    y = np.array(y_list)
    
    # 创建一个线性插值函数
    f = interp1d(x, y, kind='linear')
    
    # 计算x的最小值和最大值
    min_x = np.min(x)
    max_x = np.max(x)

    # 使用Simpson积分法计算总面积
    x_values = np.linspace(min_x, max_x, 1000)
    y_values = f(x_values)
    total_area = simps(y_values, x_values)
    
    # 计算90到最大x值的面积比
    x_90_max_values = x_values[x_values >= 90]
    y_90_max_values = f(x_90_max_values)
    area_90_max = simps(y_90_max_values, x_90_max_values)
    C90 = round(area_90_max / total_area, 3)
    
    # 计算180到最大x值的面积比
    x_180_max_values = x_values[x_values >= 180]
    y_180_max_values = f(x_180_max_values)
    area_180_max = simps(y_180_max_values, x_180_max_values)
    C180 = round(area_180_max / total_area, 3)
    
    # 绘图
    plt.figure(figsize=(16, 8))
    plt.plot(x_values, y_values, label='Interpolated Curve', color='blue')
    plt.fill_between(x_90_max_values, y_90_max_values, color='red', alpha=0.5, label='Area from 90 to Max')
    plt.fill_between(x_180_max_values, y_180_max_values, color='blue', alpha=0.5, label='Area from 180 to Max')
    plt.xlabel('Chemical shift ppm', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 设置字体为Palatino
    plt.rc('font', family='Palatino')
    
    plt.show()
    
    return C90, C180

# read json
def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# Generate substructures using DFS
def calculate_element_moles(C_moles, ele_ratio):
    '''
    Calculates the moles of each element in a compound based on the moles of carbon (C) and the mass percentage ratios of other elements relative to carbon.

    Parameters
    ----------
    C_moles : float
              The carbon counts in the compound.
    ele_ratio : dict
               Elemental percentage
    Returns
    -------
    element_moles : dict
                    Number of each element in the compound. 
    '''
    atomic_masses = {'C': 12.01, 'H': 1.008, 'O': 16.00, 'N': 14.01, 'S': 32.07}
    
    molar_ratios = {}
    for element, mass_percentage in ele_ratio.items():
        molar_ratios[element] = mass_percentage / atomic_masses[element] if mass_percentage != 0 else float('inf')

    min_ratio = min(ratio for ratio in molar_ratios.values() if ratio != float('inf'))

    relative_molar_ratios = {element: ratio / min_ratio for element, ratio in molar_ratios.items() if ratio != float('inf')}
    given_ratios = {element: round(ratio) for element, ratio in relative_molar_ratios.items()}

    rounded_relative_molar_ratios = {element: given_ratios.get(element, 0) for element in atomic_masses.keys()}

    element_moles = {}
    for element, relative_molar_ratio in rounded_relative_molar_ratios.items():
        if element == 'C':
            element_moles[element] = C_moles
        else:
            element_moles[element] = round(relative_molar_ratio * C_moles / rounded_relative_molar_ratios['C'])

    return element_moles

def merge_count_atoms(smiles):
    '''
    Calculate the target property of each element of a molecule.

    Parameters
    ----------
    smiles : string
            SMILES of a molecule
    Returns
    -------
    atom_counts: dict
                 {'C_N_ar': 15, 'C_al': 0, 'O_S': 1, 'H': 7}
    '''
    atom_counts = {'C_N_ar': 0, 'C_al': 0, 'O_S': 0, 'H': 0}
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)  # Add Hydrogens
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == 'C' or symbol == 'N':
            if symbol == 'C' and atom.GetDegree() == 4:
                atom_counts['C_al'] += 1
            else:
                atom_counts['C_N_ar'] += 1
        elif symbol == 'O' or symbol == 'S':
            atom_counts['O_S'] += 1
        if symbol == 'H':
            atom_counts['H'] += 1
    return atom_counts

def getPackage(total_smiles_list):
    '''
    Sorts a list of SMILES based on their calculated elemental compositions.
    The sorting criteria prioritize compounds with higher ratios of certain elements to H.

    Parameters
    ----------
    total_smiles_list : list
                        A list of SMILES.

    Returns
    -------
    list
        A list of SMILES strings sorted according to their elemental composition ratios.
    '''
    # Calculate the atom counts and adjust H count for each SMILES string
    adjusted_packages = [{
        'smiles': smiles,
        **merge_count_atoms(smiles),  # Unpack the atom count dictionary
        'H': merge_count_atoms(smiles)['H'] - 2  # Adjust H count
    } for smiles in total_smiles_list]

    # Sort the list of dictionaries based on element ratios
    sorted_adjusted_packages = sorted(
        adjusted_packages, 
        key=lambda package: (
            package['C_N_ar'] / (package['H'] if package['H'] else 1),  # Avoid division by zero
            package['C_al'] / (package['H'] if package['H'] else 1),
            package['O_S']
        ), 
        reverse=True
    )

    # Extract the sorted list of SMILES strings
    sorted_smiles_list = [package['smiles'] for package in sorted_adjusted_packages]

    return sorted_smiles_list

def recommended_carbonyl_hydroxyl(C_moles, ele_ratio):
    # Initial calculations
    atom_num = calculate_element_moles(C_moles, ele_ratio)
    """
    Calculate the number of carbonyl and hydroxyl groups based on atom counts and compound type.

    Parameters
    ----------
    atom_num : dict
               Atom counts for 'C', 'H', 'O', 'N', 'S'.

    Returns
    -------
    carbonyl : int
               Number of carbonyl groups.
    hydroxyl : int
               Number of hydroxyl groups.
    """
    total_mass = atom_num['C'] * 12 + atom_num['H'] * 1 + atom_num['O'] * 16 + atom_num['N'] * 14 + atom_num['S'] * 32

    carbonyl = round((0.01 * total_mass) / 16)
    hydroxyl = round((0.02 * total_mass) / 16)
    return carbonyl, hydroxyl

def getTarget(C90, ele_ratio, C_moles, carbonyl, hydroxyl):
    """
    Adjusts the target atom counts for a chemical compound based on specified carbon moles,
    elemental mass ratios, a carbon atom rate, coal type, and the percentages of carboxyl and hydroxyl groups.

    Parameters
    ----------
    C90 : float
          The degree of aromaticity.
    ele_ratio : dict
                Element symbols ('C', 'H', 'O', 'N', 'S') as keys and their mass percentages as values.
    C_moles : float
              Number of C atom.
    compound_type : str
                    Type of coal, such as 'lignite'.
    carbonyl : int
               Number of carbonyl groups.
    hydroxyl : int
               Number of hydroxyl groups.

    Returns
    -------
    atom_num : dict
               Calculate chemical formula.
    adjusted_target : dict
                      Target number of atoms.
    """
    # Initial calculations
    atom_num = calculate_element_moles(C_moles, ele_ratio)

    # Adjust targets based on carbonyl and hydroxyl values
    adjusted_target = {
        'C_N_ar': round(atom_num['C'] * C90 + atom_num['N']),
        'C_al': round(atom_num['C'] * (1 - C90)),
        'O_S': round(atom_num['O'] + atom_num['S'] - carbonyl - hydroxyl),
        'H': atom_num['H']
    }

    print(f"Model has {carbonyl} O atom in carbonyl")
    print(f"Model has {hydroxyl} O atom in hydroxyl")
    
    return atom_num, adjusted_target
    

def build_H_nested_dict(sorted_choices):
    """
    Constructs a nested dictionary to store the minimum values of hydrogen atoms (H) based on combinations of C_N_ar (carbon-nitrogen aromatic count) and O_S (oxygen-sulfur count).

    Parameters
    ----------
    sorted_choices : list of dicts
        A list where each dictionary represents a compound with keys for 'C_N_ar', 'O_S', and 'H' corresponding to carbon-nitrogen aromatic count, oxygen-sulfur count, and hydrogen count, respectively.

    Returns
    -------
    dict
        A nested dictionary where the first level keys are 'C_N_ar' values, the second level keys are 'O_S' values, and the values are the minimum 'H' values for those combinations.
    
    Example
    -------
    >>> sorted_choices = [
        {'C_N_ar': 10, 'O_S': 2, 'H': 12},
        {'C_N_ar': 10, 'O_S': 2, 'H': 8},
        {'C_N_ar': 5, 'O_S': 3, 'H': 14}
    ]
    >>> build_H_nested_dict(sorted_choices)
    {10: {2: 8}, 5: {3: 14}}
    """
    nested_dict = defaultdict(lambda: defaultdict(list))

    # Grouping H values into the nested dictionary
    for choice in sorted_choices:
        nested_dict[choice['C_N_ar']][choice['O_S']].append(choice['H'])

    # Finding the minimum H value in each list
    for c_n_ar_key in nested_dict:
        for o_s_key in nested_dict[c_n_ar_key]:
            nested_dict[c_n_ar_key][o_s_key] = min(nested_dict[c_n_ar_key][o_s_key])

    # Converting defaultdict to a standard dict
    return {c_n_ar_key: dict(inner_dict) for c_n_ar_key, inner_dict in nested_dict.items()}

def find_combinations(sorted_choices, target_C_N_ar):
    """
    Finds all unique combinations of 'C_N_ar' values from 'sorted_choices' that sum up to 'target_C_N_ar'.

    This function performs a depth-first search (DFS) to explore all possible combinations of 'C_N_ar' values that add up to a specified target. It's designed to handle and include each 'C_N_ar' value multiple times if needed to reach the target sum. The results are returned as a list of dictionaries, with each dictionary representing a unique combination where the keys are 'C_N_ar' values and the values are the counts of how many times each 'C_N_ar' is used in that combination.

    Parameters
    ----------
    sorted_choices : list of dicts
        A list of dictionaries where each dictionary contains 'C_N_ar' among other keys. 
        This list represents different choices or options, each with a specified 'C_N_ar' value.
    target_C_N_ar : int
        The target sum of 'C_N_ar' values for the combinations to reach.

    Returns
    -------
    list of dicts
        A list where each dictionary represents a unique combination of 'C_N_ar' values that sum up to 'target_C_N_ar'. The keys in each dictionary are the 'C_N_ar' values used, and the values are the counts of each 'C_N_ar' value in the combination.

    Example
    -------
    >>> sorted_choices = [
        {'C_N_ar': 2, 'O_S': 3, 'H': 7},
        {'C_N_ar': 3, 'O_S': 2, 'H': 6},
        {'C_N_ar': 5, 'O_S': 1, 'H': 8}
    ]
    >>> find_combinations(sorted_choices, 5)
    [{2: 1, 3: 1}, {5: 1}]
   """
    # Extract and sort unique 'C_N_ar' values to form candidate set
    candidates = list(set(choice['C_N_ar'] for choice in sorted_choices))
    candidates.sort()

    # Define the depth-first search (DFS) function
    def dfs(start, target, path, res):
        # Base case: if target is reached, add path to results
        if target == 0:
            res.append(path)
            return
        # Stop searching if target is negative or no more candidates left
        if target < 0 or start == len(candidates):
            return
        # Avoid using 0 candidate and continue DFS with the next candidate
        if candidates[start] == 0:
            dfs(start + 1, target, path, res)
            return
        # Include current candidate and continue DFS
        dfs(start, target - candidates[start], path + [candidates[start]], res)
        # Exclude current candidate and continue DFS with the next candidate
        dfs(start + 1, target, path, res)

    # Initialize results list and start DFS
    res = []
    dfs(0, target_C_N_ar, [], res)

    # Convert each combination from list to dictionary format
    cn_list = []
    for combination in res:
        temp_dict = {key: combination.count(key) for key in candidates}
        cn_list.append(temp_dict)
    return cn_list

def backtrack_combinations(nested_dict_H, selection_dic, target_O_S, max_depth=30):
    """
    Finds combinations within a nested dictionary that match a specified target for oxygen and sulfur count (O_S) using backtracking.

    Parameters
    ----------
    nested_dict_H : dict
        A nested dictionary where the first level keys are H (hydrogen) counts, and the second level keys are O_S (oxygen and sulfur) counts with their respective counts as values.
    selection_dic : dict
        A dictionary indicating the total counts for each H key that must be met in the solution.
    target_O_S : int
        The target sum of O_S values that the combinations should match.
    max_depth : int, optional
        The maximum depth to which the backtracking algorithm will explore. Default is 30.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a valid combination that matches the target_O_S and adheres to the counts specified in selection_dic.
   """
    # Sort H keys in descending order for optimization
    H_keys = sorted(nested_dict_H.keys(), reverse=True)
    solutions = []
    memo = {}

    def convert_to_hashable(current_selection):
        """
        Converts the current selection dictionary into a hashable type (tuple) for memoization.
        """
        return tuple((k, tuple(v.items())) for k, v in current_selection.items())

    def backtrack(remaining_O_S, H_index, current_selection, depth):
        """
        Recursive backtracking function to explore combinations.
        """
        # Base case: Stop if maximum depth is exceeded
        if depth > max_depth:
            return
        
        # Check memo to avoid re-exploring known paths
        hashable_key = (remaining_O_S, H_index, convert_to_hashable(current_selection))
        if hashable_key in memo:
            return

        # Successful case: If target O_S is met and all H counts match selection criteria
        if remaining_O_S == 0 and all(sum(current_selection[H].values()) == selection_dic[H] for H in H_keys):
            solutions.append(copy.deepcopy(current_selection))
            return
        
        # Base case: Stop if index out of bounds or negative target
        if H_index >= len(H_keys) or remaining_O_S < 0:
            return
        
        current_H = H_keys[H_index]
        # Explore next H value without adding current H
        backtrack(remaining_O_S, H_index + 1, current_selection, depth + 1)

        # Try adding current H and explore further
        for O_S, count in nested_dict_H[current_H].items():
            if sum(current_selection[current_H].values()) < selection_dic[current_H]:
                current_selection[current_H][O_S] += 1
                backtrack(remaining_O_S - O_S, H_index, current_selection, depth + 1)
                current_selection[current_H][O_S] -= 1

        memo[hashable_key] = True
                    
    # Initialize the selection with zeros for each possible O_S count under each H key
    initial_selection = {key: {subkey: 0 for subkey in nested_dict_H[key]} for key in nested_dict_H.keys()}
    backtrack(target_O_S, 0, initial_selection, 0)  # Start backtracking with initial conditions

    return solutions


def parallel_backtrack_combinations(nested_dict_H, selection_dic, target_O_S, max_depth=50, n_jobs=40):
    """
    Parallelized version of backtracking to find combinations that match target_O_S.

    Parameters
    ----------
    nested_dict_H : dict
        A nested dictionary where the first level keys are H (hydrogen) counts.
    selection_dic : dict
        A dictionary specifying total counts for each H key in the solution.
    target_O_S : int
        The target sum of O_S values to match.
    n_jobs : int, optional
        Number of parallel jobs (cores) to use.

    Returns
    -------
    list
        A list of dictionaries, each representing a valid combination.
    """
    # Sort H keys in descending order for optimization
    H_keys = sorted(nested_dict_H.keys(), reverse=True)

    def convert_to_hashable(current_selection):
        """
        Converts the current selection dictionary into a hashable type (tuple) for memoization.
        """
        return tuple((k, tuple(v.items())) for k, v in current_selection.items())

    def backtrack(remaining_O_S, H_index, current_selection, depth, memo):
        """
        Recursive backtracking function to explore combinations.
        """
        # Base case: Stop if maximum depth is exceeded
        if depth > max_depth:  # Max depth hardcoded here
            return []

        # Check memo to avoid re-exploring known paths
        hashable_key = (remaining_O_S, H_index, convert_to_hashable(current_selection))
        if hashable_key in memo:
            return []

        # Successful case
        if remaining_O_S == 0 and all(
            sum(current_selection[H].values()) == selection_dic[H] for H in H_keys
        ):
            return [copy.deepcopy(current_selection)]

        # Base case: Stop if index out of bounds or negative target
        if H_index >= len(H_keys) or remaining_O_S < 0:
            return []

        current_H = H_keys[H_index]
        solutions = []

        # Explore next H value without adding current H
        solutions.extend(backtrack(remaining_O_S, H_index + 1, current_selection, depth + 1, memo))

        # Try adding current H and explore further
        for O_S, count in nested_dict_H[current_H].items():
            if sum(current_selection[current_H].values()) < selection_dic[current_H]:
                current_selection[current_H][O_S] += 1
                solutions.extend(
                    backtrack(remaining_O_S - O_S, H_index, current_selection, depth + 1, memo)
                )
                current_selection[current_H][O_S] -= 1

        memo[hashable_key] = True
        return solutions

    def generate_subtasks(nested_dict_H):
        """
        Generate finer-grained subtasks for parallelization.
        """
        subtasks = []
        for H_key, sub_dict in nested_dict_H.items():
            for O_S_key in sub_dict.keys():
                subtasks.append((H_key, O_S_key))
        return subtasks

    def backtrack_for_subtask(H_key, O_S_key):
        """
        Perform backtracking for a specific H and O_S combination.
        """
        memo = {}
        initial_selection = {
            key: {subkey: 0 for subkey in nested_dict_H[key]} for key in nested_dict_H
        }
        initial_selection[H_key][O_S_key] += 1  # Initialize with one choice
        remaining_O_S = target_O_S - O_S_key
        return backtrack(remaining_O_S, H_keys.index(H_key), initial_selection, 0, memo)

    # Generate finer-grained subtasks
    subtasks = generate_subtasks(nested_dict_H)

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(backtrack_for_subtask)(H_key, O_S_key) for H_key, O_S_key in subtasks
    )

    # Combine results
    combined_solutions = []
    for result in results:
        combined_solutions.extend(result)

    return combined_solutions


def find_matching_indices(selection, sorted_choices):
    """
    Finds indices in 'sorted_choices' that match the criteria specified in 'selection',
    considering constraints on repetition.

    Parameters
    ----------
    selection : dict
        A nested dictionary where the first level keys are 'C_N_ar' values, and the second level keys are 'O_S' values, each with an integer indicating how many times that combination should be selected.
    sorted_choices : list of dicts
        A list where each dictionary represents a possible choice with 'C_N_ar' and 'O_S' keys among potentially others.

    Returns
    -------
    indices : list
             A list of indices from 'sorted_choices' that match the 'selection' criteria. 
             Each index corresponds to an entry in 'sorted_choices' that fits one of the combinations specified in 'selection' and is chosen considering the maximum repetition allowed.
    """
    indices = []
    max_repeat_times = {}  # Tracks the number of times each index is selected

    for key in selection:
        for subkey in selection[key]:
            if selection[key][subkey] > 0:
                # Identify all indices that match the current 'C_N_ar' and 'O_S' criteria
                matches = [i for i, choice in enumerate(sorted_choices) 
                           if choice['C_N_ar'] == key and choice['O_S'] == subkey]
                # Calculate the maximum allowed repeats for each index to distribute selections evenly
                if matches:
                    max_repeats = math.ceil(selection[key][subkey] / len(matches))
                    for match in matches:
                        max_repeat_times[match] = max_repeats

                # Randomly select among the matching indices, considering repeat limits
                for _ in range(selection[key][subkey]):
                    valid_matches = [index for index in matches if max_repeat_times[index] > 0]
                    if valid_matches:
                        selected = random.choice(valid_matches)
                        indices.append(selected)
                        # Decrease the allowed repeats for the selected index
                        max_repeat_times[selected] -= 1

    return indices

def generate_candidate_smiles(min_H_selection, sorted_choices, sorted_smiles_list, target_C_al):
    """
    Generates a list of candidate SMILES strings based on a selection criteria, and calculates the predicted chemical formula for these candidates, adjusting for a target count of aliphatic carbon (C_al).

    Parameters
    ----------
    min_H_selection : dict
        A selection criteria specifying the minimum hydrogen count selections for candidates.
    sorted_choices : list of dicts
        A list of dictionaries, each representing a choice with details including 'C_N_ar', 'C_al', 'O_S', and 'H' counts.
    sorted_smiles_list : list of str
        A list of SMILES strings corresponding to the choices in `sorted_choices`.
    target_C_al : int
        The target total count of aliphatic carbon atoms in the combined candidates.

    Returns
    -------
    tuple
        - A list of selected candidate SMILES strings that meet the selection criteria.
        - A string representing the predicted chemical formula of the combined candidates, adjusting for any additional carbon atoms to meet the target C_al count.
    """
    indices = find_matching_indices(min_H_selection, sorted_choices)
    candidate_smiles_list = [sorted_smiles_list[i] for i in indices]
    selected_legends = [str(sorted_choices[i]) for i in indices]

    # Extract values for C_N_ar, C_al, and O_S from the selection legends
    C_al_values = [ast.literal_eval(legend)['C_al'] for legend in selected_legends]
    total_C_al = sum(C_al_values)
    additional_C_al = target_C_al - total_C_al
    added_value = max(additional_C_al, 0)

    # Additional extractions are redundant as values are recalculated without being used again
    # Calculation of total counts is directly followed by the adjustment for extra carbons
    for _ in range(added_value):
        candidate_smiles_list.append('C')

    element_counts = count_atoms(candidate_smiles_list)

    return candidate_smiles_list, element_counts

def find_min_H_selection(self, all_final_selections):
    """
    Finds the selection from all possible combinations that results in the minimum total H count.

    Parameters
    ----------
    all_final_selections : list
        A list of dictionaries, each representing a final selection of combinations with their respective counts.

    Returns
    -------
    dict
        The selection (a dictionary) from `all_final_selections` that has the minimum total H count.
    """
    # Initialize variables to track the minimum H count and the corresponding selection
    min_total_H = float('inf')
    min_H_selection = None

    # Iterate through each selection to calculate its total H count
    for selection in all_final_selections:
        total_H = 0
        # Iterate through each H value in the selection
        for H in selection:
            # Iterate through each O_S value under the current H and accumulate the total H
            for O_S, count in selection[H].items():
                total_H += count * self.nested_dict_H[H][O_S]

        # If the current selection's total H is less than the minimum found so far, update min_total_H and min_H_selection
        if total_H < min_total_H:
            min_total_H = total_H
            min_H_selection = selection

    return min_H_selection


# Generate final coal from substructures

def show_atom_number(mol, label):
    """
    Label each atom in the given molecule with its index.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object whose atoms are to be labeled. This object is modified in place.
    label : str
        The label under which the atom index will be stored. This label can then be used to access the atom index from the atom properties.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The RDKit molecule object with atoms labeled with their indices. Note that the molecule is modified in place, so the returned object is the same as the input object.
    """
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

def find_required_aldehyde_carbons(smiles):
    """
    Identifies carbon atoms in a molecule that are part of an aldehyde group. 
    An aldehyde carbon is defined as a carbon atom having exactly three neighbors: 
    one carbon (C), one hydrogen (H), and one oxygen (O).

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that are part of an aldehyde group within the molecule.
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
    """
    # Create a molecule object from SMILES and add hydrogens explicitly
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # Initialize a list to store indices of carbon atoms that are part of an aldehyde group
    carbons = []

    # Iterate through each atom in the molecule
    for atom in mol.GetAtoms():
        # Check if the atom is a carbon
        if atom.GetSymbol() == 'C':
            # Get neighboring atoms
            neighbors = atom.GetNeighbors()
            
            # Check for exactly three neighbors: one C, one H, and one O
            if len(neighbors) == 3:
                symbols = [neighbor.GetSymbol() for neighbor in neighbors]
                if symbols.count('C') == 1 and symbols.count('H') == 1 and symbols.count('O') == 1:
                    # If conditions are met, append the carbon atom's index to the list
                    carbons.append(atom.GetIdx())

    # Return the list of carbon atom indices that are part of an aldehyde group
    return carbons

def find_beta_carbons(smiles):
    """
    Identifies pairs of aldehyde (carbonyl) carbon atoms and their beta carbon atoms in a molecule.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure.

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains two integers. The first integer is the index of an aldehyde carbon atom, 
        and the second is the index of a corresponding beta carbon atom that is bonded to at least one hydrogen atom.
        Atom indices are zero-based and correspond to the order in which atoms appear in the RDKit molecule object.
    """
    # Get indices of aldehyde carbon atoms
    aldehyde_carbons = find_required_aldehyde_carbons(smiles)

    # Create a molecule object from SMILES and add hydrogens explicitly
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
    # Initialize a list to store pairs of aldehyde and beta carbon atoms
    carbon_pairs = []

    # Iterate through each aldehyde carbon atom
    for aldehyde_carbon in aldehyde_carbons:
        # Find alpha carbon atoms (directly bonded carbon atoms)
        alpha_carbon = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(aldehyde_carbon).GetNeighbors() if neighbor.GetSymbol() == 'C']
        
        all_beta_carbons = []

        # Iterate through each alpha carbon to find beta carbon atoms
        for alpha in alpha_carbon:
            # Find beta carbon atoms that are bonded to at least one hydrogen atom
            beta_carbons = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(alpha).GetNeighbors() if neighbor.GetSymbol() == 'C' and any(n.GetSymbol() == 'H' for n in neighbor.GetNeighbors())]

            all_beta_carbons.extend(beta_carbons)

        # Ensure beta carbon is not the aldehyde carbon itself
        all_beta_carbons = [beta for beta in all_beta_carbons if beta != aldehyde_carbon]

        # Randomly select a beta carbon for pairing if available
        if all_beta_carbons:
            chosen_beta_carbon = random.choice(all_beta_carbons)
            carbon_pairs.append((aldehyde_carbon, chosen_beta_carbon))

    return carbon_pairs

def connect_rings_C3(smiles1):
    """
    Connects a given molecule with a propane molecule at identified beta carbon positions.

    Parameters
    ----------
    smiles1 : str
        A SMILES string representing the initial molecule to which propane will be connected.

    Returns
    -------
    str or None
        A SMILES string representing the modified molecule after connection with propane. 
        If no suitable carbon atoms are found for the connection, returns None.
    """
    smiles2 = 'CCC'
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    carbons1_list = find_beta_carbons(smiles1)

    if not carbons1_list:
        print("No suitable aldehyde carbon atom found in the molecule")
        return None

    for chosen_pair1 in carbons1_list:
        combined = Chem.CombineMols(mol1, mol2)
        edit_combined = Chem.EditableMol(combined)

        # Calculate new indices in the combined molecule
        index1, index2 = chosen_pair1
        index3 = len(mol1.GetAtoms())
        index4 = index3 + 2

        # Connect the molecules at the calculated indices
        edit_combined.AddBond(index1, index3, order=BondType.SINGLE)
        edit_combined.AddBond(index2, index4, order=BondType.SINGLE)

        mol1 = edit_combined.GetMol()
        Chem.SanitizeMol(mol1)

    return Chem.MolToSmiles(mol1)

def update_smiles_lists(smiles_list1, smiles_list2):
    """
    Processes two lists of SMILES strings: connects propane molecules to the first list's molecules if specific criteria are met, 
    and then adjusts the second list by removing elements to compensate for the carbons added to the first list.

    Parameters
    ----------
    smiles_list1 : list of str
        The first list of SMILES strings to be potentially modified by connecting propane molecules.
    smiles_list2 : list of str
        The second list of SMILES strings to be adjusted by removing elements based on the number of carbons added to the first list.

    Returns
    -------
    tuple of lists or None
        Returns a tuple containing two lists:
        - The modified first list with propane molecules connected as applicable.
        - The adjusted second list with elements removed to compensate for added carbons.
        If there are not enough elements in the second list to compensate for the added carbons, returns None.
   """
    new_smiles_list1 = []
    carbon_diff = 0
    
    for smiles in smiles_list1:
        if find_required_aldehyde_carbons(smiles):
            new_smiles = connect_rings_C3(smiles)
        else:
            new_smiles = smiles
        new_smiles_list1.append(new_smiles)

        old_carbons = smiles.count('C')
        new_carbons = new_smiles.count('C')
        
        carbon_diff += new_carbons - old_carbons

    if carbon_diff > len(smiles_list2):
        print(f"Not enough carbons in the second list to compensate the increase in the first list. Needed: {carbon_diff}, Available: {len(smiles_list2)}")
        return None

    new_smiles_list2 = smiles_list2[carbon_diff:]
    
    return new_smiles_list1, new_smiles_list2


def find_required_carbons(smiles):
    """
    Identifies carbon atoms in a molecule that are connected to exactly two other carbon (or nitrogen) atoms and one hydrogen atom.
    
    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure to be analyzed.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the specified connectivity criteria within the molecule.
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
    """
    # Create an RDKit molecule object from the SMILES string and add hydrogens explicitly
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    
    # Initialize a list to store indices of carbon atoms meeting the criteria
    required_carbons = []

    # Iterate over all atoms in the molecule
    for atom in molecule.GetAtoms():
        # Check if the atom is a carbon atom
        if atom.GetSymbol() == "C":
            # Get all neighboring atoms of the current carbon atom
            neighbors = atom.GetNeighbors()
            
            # Count the number of carbon (or nitrogen) neighbors
            carbon_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() == 'C' or neighbor.GetSymbol() == 'N'])

            # Count the number of hydrogen neighbors
            hydrogen_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() == 'H'])

            # If the carbon atom is connected to exactly two carbon (or nitrogen) atoms and one hydrogen atom
            if carbon_neighbors == 2 and hydrogen_neighbors == 1:
                # Add the index of the carbon atom to the list
                required_carbons.append(atom.GetIdx())
    
    # Return the list of indices of carbon atoms meeting the criteria
    return required_carbons

def _find_adjacent_pairs(carbons, molecule):
    """
    Identifies pairs of adjacent carbon atoms suitable for connection.
    
    Parameters
    ----------
    carbons : list of int
        Indices of carbon atoms considered for forming pairs.
    molecule : rdkit.Chem.rdchem.Mol
        The molecule object containing the carbon atoms.
    
    Returns
    -------
    list of tuples
        A list containing tuples, each with two integers representing the indices of adjacent carbon atoms suitable for connection.
    """
    index_pairs = []
    for i in range(len(carbons)):
        atom = molecule.GetAtomWithIdx(carbons[i])
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() in carbons:
                if carbons[i] < neighbor.GetIdx():  # Prevent duplicates
                    index_pairs.append((carbons[i], neighbor.GetIdx()))
    return index_pairs

def _connect_ring_C(mol1, mol2, chosen_pair1, chosen_pair2):
    """
    Connects two molecules using two methane molecules to simulate the addition of methyl groups at specified carbon atoms.

    Parameters
    ----------
    mol1, mol2 : RDKit Mol
        The molecule objects to be connected.
    chosen_pair1, chosen_pair2 : tuple
        Pairs of carbon atom indices in mol1 and mol2, respectively, where the methane molecules will be connected.

    Returns
    -------
    RDKit Mol
        A new RDKit Mol object that represents the connected molecule structure.
    """
    methane1 = Chem.MolFromSmiles('C')
    methane2 = Chem.MolFromSmiles('C')
    
    combined_mol = Chem.CombineMols(mol1, methane1)
    combined_mol = Chem.CombineMols(combined_mol, methane2)
    edit_mol = Chem.EditableMol(combined_mol)
    edit_mol.AddBond(chosen_pair1[0], len(mol1.GetAtoms()), BondType.SINGLE)
    edit_mol.AddBond(chosen_pair1[1], len(mol1.GetAtoms()) + 1, BondType.SINGLE)

    connected_mol = edit_mol.GetMol()
    Chem.SanitizeMol(connected_mol)

    methane1_C_index = len(mol1.GetAtoms())
    methane2_C_index = len(mol1.GetAtoms()) + 1

    combined_mol2 = Chem.CombineMols(mol2, connected_mol)
    edit_mol2 = Chem.EditableMol(combined_mol2)
    edit_mol2.AddBond(chosen_pair2[0], len(mol2.GetAtoms()) + methane1_C_index, BondType.SINGLE)
    edit_mol2.AddBond(chosen_pair2[1], len(mol2.GetAtoms()) + methane2_C_index, BondType.SINGLE)

    final_connected_mol = edit_mol2.GetMol()
    Chem.SanitizeMol(final_connected_mol)

    return final_connected_mol

def _find_index_pairs(carbons, mol):
    """
    Finds index pairs of carbon atoms within a molecule for connection.

    Parameters
    ----------
    carbons : list of int
        A list containing indices of carbon atoms within the molecule that are suitable for connection.
    mol : RDKit Mol object
        The molecule in which the carbon atoms are located.

    Returns
    -------
    list of tuples
        A list where each tuple contains two indices representing a pair of carbon atoms.
    """
    index_pairs = []
    for idx in carbons:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in carbons:
                neighbor_idx = neighbor.GetIdx()
                if idx < neighbor_idx:  # Avoid repeating the same pair in reverse order
                    index_pairs.append((idx, neighbor_idx))
    return index_pairs


def connect_rings(molecules_tuple_list):
    """
    Connects two rings from a list of molecule tuples by adding two methane groups between them.

    Parameters
    ----------
    molecules_tuple_list : list of tuples
        A list where each tuple contains a SMILES string of a molecule and a list of carbon atom indices.

    Returns
    -------
    SMILES
        A new SMILES string that represents the connected molecule structure.
    """
    if len(molecules_tuple_list) < 2:
        # 如果列表中只有一个元组，直接返回该元组中的SMILES字符串
        if len(molecules_tuple_list) == 1:
            return molecules_tuple_list[0][0]
        # 如果列表为空或不足，打印出问题的输入
        print(f"Invalid input: {molecules_tuple_list}")
        return "Invalid input. The list must contain at least one molecule."

    smiles1, carbons1 = molecules_tuple_list[0]
    smiles2, carbons2 = molecules_tuple_list[1]

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    index_pairs_carbons1 = _find_index_pairs(carbons1, mol1)
    index_pairs_carbons2 = _find_index_pairs(carbons2, mol2)

    chosen_pair1 = random.choice(index_pairs_carbons1)
    chosen_pair2 = random.choice(index_pairs_carbons2)

    # Call the new function to connect the molecules with methane
    final_connected_mol = _connect_ring_C(mol1, mol2, chosen_pair1, chosen_pair2)

    final_connected_smiles = Chem.MolToSmiles(final_connected_mol)
    return final_connected_smiles


def count_hydroxy_oxygen(smiles_list):
    """
    Counts the total number of hydroxy oxygen and sulfur atoms in a list of molecules represented by SMILES strings.

    A hydroxy oxygen or sulfur atom is defined as an oxygen or sulfur atom that is bonded to at least one hydrogen atom, 
    indicative of alcohol, phenol, or thiol groups.

    Parameters
    ----------
    smiles_list : list of str
        A list containing SMILES strings of the molecules to analyze.

    Returns
    -------
    total_count : int
                  The total count of hydroxy oxygen and sulfur atoms across all molecules in the given list.
    """
    total_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_with_hs = AddHs(mol)  # Explicitly add hydrogens to the molecule.

        hydroxy_atoms = []
        for atom in mol_with_hs.GetAtoms():
            if atom.GetSymbol() in ['O', 'S']:  # Check for oxygen and sulfur atoms.
                neighbors = atom.GetNeighbors()
                has_hydrogen = False
                for neighbor in neighbors:
                    if neighbor.GetSymbol() == 'H':
                        has_hydrogen = True
                        break
                if has_hydrogen:
                    hydroxy_atoms.append(atom.GetIdx())  # Append index of hydroxy oxygen or sulfur atoms.
        total_count += len(hydroxy_atoms)
    return total_count

def count_ketone_carbons(smiles_list):
    """
    Counts the total number of ketone carbon atoms in a list of molecules represented by SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        A list containing SMILES strings of the molecules to analyze.

    Returns
    -------
    total_count : int
                  The total count of ketone carbon atoms across all molecules in the given list.
    """
    total_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_with_hs = Chem.AddHs(mol)  # Add hydrogens explicitly.

        ketone_carbons = []
        for atom in mol_with_hs.GetAtoms():
            if atom.GetSymbol() == 'C':
                neighbors = atom.GetNeighbors()
                has_double_bonded_oxygen = False
                for neighbor in neighbors:
                    # Check for a double bond between the carbon and an oxygen atom.
                    if neighbor.GetSymbol() == 'O' and mol_with_hs.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == rdchem.BondType.DOUBLE:
                        has_double_bonded_oxygen = True
                        break
                if has_double_bonded_oxygen:
                    ketone_carbons.append(atom.GetIdx())  # Add the index of the ketone carbon atom.
        total_count += len(ketone_carbons)
    return total_count

def find_C4_carbons(smiles):
    """
    Identifies carbon atoms in a molecule that are bonded to exactly two carbon or nitrogen atoms and have one or two hydrogen atoms.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure to be analyzed.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the specified bonding criteria within the molecule. Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Convert SMILES to molecule and explicitly add hydrogens.

    required_carbons = []  # Initialize a list to store indices of carbon atoms meeting the criteria.

    # Iterate through all atoms in the molecule.
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "C":  # Check if the atom is a carbon atom.
            neighbors = atom.GetNeighbors()  # Get all neighboring atoms.

            # Count carbon and nitrogen neighbors.
            carbon_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() in ['C', 'N']])

            # Count hydrogen neighbors.
            hydrogen_neighbors = len([neighbor for neighbor in neighbors if neighbor.GetSymbol() == 'H'])

            # Check if the carbon atom meets the specified bonding criteria.
            if carbon_neighbors == 2 and (hydrogen_neighbors == 1 or hydrogen_neighbors == 2):
                required_carbons.append(atom.GetIdx())  # Add the carbon atom's index to the list.

    return required_carbons  # Return the list of indices of carbon atoms meeting the criteria.

def connect_rings_C4(smiles1):
    """
    Connects a given molecule with a butane molecule at specified carbon atom positions.
    
    Parameters
    ----------
    smiles1 : str
        A SMILES string representing the initial molecule to which butane will be connected.

    Returns
    -------
    str or None
        A SMILES string representing the modified molecule after connection with butane. 
        If no suitable carbon atoms are found for the connection, returns None.
    """
    smiles2 = 'CCCC'
    carbons2 = [0, 3]  # Butane's terminal carbon indices, not used directly but illustrative for extension purposes

    carbons1 = find_C4_carbons(smiles1)

    if not carbons1:
        print("No suitable carbon atom found in the molecule")
        return None

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Identify adjacent carbon pairs in the first molecule for connection
    index_pairs_carbons1 = _find_adjacent_pairs(carbons1, mol1)

    if not index_pairs_carbons1:
        print(f"No suitable pair of carbon atoms found in the molecule: {smiles1}")
        return None

    chosen_pair1 = random.choice(index_pairs_carbons1)

    combined = Chem.CombineMols(mol1, mol2)
    edit_combined = Chem.EditableMol(combined)

    # Recalculate indices in the combined molecule
    index1, index2 = chosen_pair1  # Indices for carbons in mol1
    index3 = len(mol1.GetAtoms())  # Index for the first atom of mol2 in the combined molecule
    index4 = len(mol1.GetAtoms()) + 3  # Index for the last atom of mol2 in the combined molecule

    # Add bonds between the selected carbons and butane's terminal carbons
    edit_combined.AddBond(index1, index3, order=BondType.SINGLE)
    edit_combined.AddBond(index2, index4, order=BondType.SINGLE)

    connected_mol = edit_combined.GetMol()
    Chem.SanitizeMol(connected_mol)

    return Chem.MolToSmiles(connected_mol)

def repeat_connect_rings_C4(smiles, num_repeats):
    """
    Iteratively connects a butane molecule to the given molecule a specified number of times.
    
    Parameters
    ----------
    smiles : str
        A SMILES string representing the initial molecule to be modified.
    num_repeats : int
        The number of times the butane molecule should be connected to the initial molecule.
        
    Returns
    -------
    str
        The SMILES string representing the modified molecule after all connections have been made.
        If at any point the connection cannot be made (e.g., no suitable carbons are found), the function
        returns the most recent successful modification or the original molecule if none were made.
    """
    for _ in range(num_repeats):
        new_smiles = connect_rings_C4(smiles)
        if new_smiles is None:
            break  # Stop if no suitable carbon atom found or connection failed
        smiles = new_smiles
    return smiles

def process_smiles(smiles_list):
    """
    Sorts a list of SMILES strings in a zigzag pattern based on the sum of atomic numbers in each molecule.

    Parameters
    ----------
    smiles_list : list of str
        A list containing SMILES strings of the molecules to analyze.

    Returns
    -------
    list of str
        The list of SMILES strings sorted in a zigzag pattern based on their molecular weight.
    """
    # Inline function to count atomic numbers in a molecule
    def count_atoms_inline(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return sum(atom.GetAtomicNum() for atom in mol.GetAtoms())

    # Sort the SMILES strings based on their atomic numbers
    sorted_items = sorted(smiles_list, key=count_atoms_inline)
    
    # Initialize the result list with None to match the input list's length
    result = [None] * len(smiles_list)
    
    # Apply the zigzag sorting logic
    start, end = 0, len(sorted_items) - 1
    for index in range(len(sorted_items)):
        if index % 2 == 0:
            result[index] = sorted_items[end]
            end -= 1
        else:
            result[index] = sorted_items[start]
            start += 1
    
    return result

def find_aldehyde_carbons(mol):
    """
    Identifies aldehyde carbon atoms within a given molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object to be analyzed.

    Returns
    -------
    aldehyde_carbons : list
        A list of indices corresponding to carbon atoms that are part of an aldehyde group within the molecule.
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
    """
    mol_with_hs = Chem.AddHs(mol)  # Explicitly add all hydrogens to the molecule for accurate analysis

    aldehyde_carbons = []
    for atom in mol_with_hs.GetAtoms():
        if atom.GetSymbol() == 'C':
            oxygen_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O']
            hydrogen_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'H']
            
            # Check for one oxygen neighbor, one hydrogen neighbor, and the presence of a double bond with oxygen
            if len(oxygen_neighbors) == 1 and len(hydrogen_neighbors) == 1:
                for bond in atom.GetBonds():
                    if bond.GetOtherAtom(atom).GetSymbol() == 'O' and bond.GetBondType() == rdchem.BondType.DOUBLE:
                        aldehyde_carbons.append(atom.GetIdx())
                        break  # Once an aldehyde group is confirmed, no need to check other bonds of the same carbon

    return aldehyde_carbons

def find_hydroxy_oxygen(mol):
    """
    Identifies the indices of hydroxy oxygen and sulfur atoms in a given RDKit molecule object.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object to be analyzed.

    Returns
    -------
    hydroxy_atoms : list
        A list of indices for oxygen and sulfur atoms bonded to at least one hydrogen atom,
        indicating the presence of hydroxy or thiol groups respectively.
    """
    mol_with_hs = AddHs(mol)  # Add explicit hydrogens to the molecule for accurate analysis.

    hydroxy_atoms = []
    for atom in mol_with_hs.GetAtoms():
        if atom.GetSymbol() in ['O', 'S']:  # Check for oxygen and sulfur atoms.
            # Check if the atom is bonded to a hydrogen atom.
            has_hydrogen = any(neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors())
            if has_hydrogen:
                hydroxy_atoms.append(atom.GetIdx())  # Append the index of the hydroxy atom.

    return hydroxy_atoms

def find_ketone_alpha_carbons(mol):
    """
    Identifies alpha carbon atoms adjacent to carbonyl carbons in ketones within a given molecule.

    Parameters
    ----------
    mol : RDKit Mol
        An RDKit molecule object.

    Returns
    -------
    ketone_alpha_carbons : list
        A list of indices for alpha carbon atoms adjacent to carbonyl carbons in ketones.
   """
    ketone_alpha_carbons = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'O':
            if atom.GetTotalNumHs() == 0:  # Check for oxygen atoms not bonded to any hydrogens
                neighbors = atom.GetNeighbors()
                non_H_neighbors = [x for x in neighbors if x.GetSymbol() != 'H']
                if len(non_H_neighbors) == 1:
                    carbonyl_carbon = non_H_neighbors[0]
                    # Ensure the bond between oxygen and the carbonyl carbon is a double bond
                    if mol.GetBondBetweenAtoms(atom.GetIdx(), carbonyl_carbon.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
                        # Now find the alpha carbons: those bonded to the carbonyl carbon, excluding the oxygen itself
                        for neighbor in carbonyl_carbon.GetNeighbors():
                            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() != atom.GetIdx():
                                ketone_alpha_carbons.append(neighbor.GetIdx())

    return ketone_alpha_carbons

def find_alpha_carbons(mol):
    """
    Identifies indices of non-aromatic alpha carbon atoms that are directly bonded to aromatic carbon atoms.

    Parameters
    ----------
    mol : RDKit Mol
        An RDKit molecule object to be analyzed.

    Returns
    -------
    list(alpha_carbons) : list
        A list of unique indices for non-aromatic alpha carbon atoms directly bonded to aromatic carbon atoms.
    """
    alpha_carbons = set()  # Use a set to avoid duplicate indices

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetIsAromatic():
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C' and not neighbor.GetIsAromatic():
                    alpha_carbons.add(neighbor.GetIdx())  # Add unique alpha carbon indices

    return list(alpha_carbons)  # Convert back to list for consistency with expected return type

def find_aliphatic_carbons(mol):
    """
    Identifies indices of aliphatic (non-aromatic) carbon atoms within a given molecule.

    Parameters
    ----------
    mol : RDKit Mol
        An RDKit molecule object to be analyzed.

    Returns
    -------
    aliphatic_carbons : list
        A list of indices corresponding to aliphatic carbon atoms in the molecule.
    """
    aliphatic_carbons = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
            aliphatic_carbons.append(atom.GetIdx())  # Add the index of each aliphatic carbon atom

    return aliphatic_carbons

def find_benzene_carbons(mol):
    """
    Identifies indices of carbon atoms that are part of benzene rings in a given molecule.

    Parameters
    ----------
    mol : RDKit Mol
        An RDKit molecule object to be analyzed.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that are part of benzene rings.
    """
    benzene_carbons = []
    ri = mol.GetRingInfo()
    aromatic_rings = [ring for ring in ri.AtomRings() if len(ring) == 6 and all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring)]

    for ring in aromatic_rings:
        for idx in ring:
            if mol.GetAtomWithIdx(idx).GetIsAromatic():
                benzene_carbons.append(idx)

    return list(set(benzene_carbons))  # Remove duplicates and return the list

def find_ban_carbons(mol):
    """
    Identifies carbon atoms bonded to exactly two hydrogen atoms and not connected to any other carbons from the list.

    Parameters
    ----------
    mol : RDKit Mol
        An RDKit molecule object before adding explicit hydrogens.

    Returns
    -------
    single_linked_carbons : list
                            A list of indices for carbon atoms that meet the criteria.
    """
    # Explicitly add hydrogens to the molecule for accurate processing
    mol_with_hs = AddHs(mol)

    # Identify carbon atoms bonded to exactly two hydrogen atoms
    carbon_atoms_with_two_hydrogen = []
    for atom in mol_with_hs.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() == 2:
            carbon_atoms_with_two_hydrogen.append(atom.GetIdx())

    # Determine neighbor atoms for each carbon atom identified above
    neighbor_atoms = {
        idx: [a.GetIdx() for a in mol_with_hs.GetAtomWithIdx(idx).GetNeighbors() if a.GetSymbol() != 'H']
        for idx in carbon_atoms_with_two_hydrogen
    }

    # Find carbon atoms only connected to hydrogen atoms (excluding those connected to other carbons in the list)
    single_linked_carbons = [
        idx for idx, neighbors in neighbor_atoms.items()
        if not any(neighbor in carbon_atoms_with_two_hydrogen for neighbor in neighbors)
    ]
    
    return single_linked_carbons

def connect_molecules(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # Define lists of functions to identify suitable connection points on each molecule
    connection_functions1 = [
        find_aldehyde_carbons,
        find_hydroxy_oxygen,
        find_ketone_alpha_carbons,
        find_alpha_carbons, 
        find_aliphatic_carbons,
        find_benzene_carbons
    ]
    
    connection_functions2 = [
        find_aldehyde_carbons,
        find_hydroxy_oxygen,
        find_ketone_alpha_carbons,
        find_alpha_carbons,
        find_aliphatic_carbons,
        find_benzene_carbons
    ]
       
    for find_atoms1 in connection_functions1:
        atoms1 = find_atoms1(mol1)
        atoms1 = [idx for idx in atoms1 if len([neighbor for neighbor in mol1.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetSymbol() != 'H']) < 3]
        atoms1 = [idx for idx in atoms1 if idx not in find_ban_carbons(mol1)]
        
        if atoms1:
            for find_atoms2 in connection_functions2:
                atoms2 = find_atoms2(mol2)
                atoms2 = [idx for idx in atoms2 if len([neighbor for neighbor in mol2.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetSymbol() != 'H']) < 3]
                atoms2 = [idx for idx in atoms2 if idx not in find_ban_carbons(mol2)]
                
                if atoms2 and not (find_atoms1 == find_hydroxy_oxygen and find_atoms2 == find_hydroxy_oxygen):
                    try:
                        atom1_idx = random.choice(atoms1)
                        atom2_idx = random.choice(atoms2)
                        new_mol = Chem.CombineMols(mol1, mol2)
                        edit_mol = Chem.EditableMol(new_mol)
                        # Correct index for atom2 after combining molecules
                        combined_atom2_idx = len(mol1.GetAtoms()) + atom2_idx
                        edit_mol.AddBond(atom1_idx, combined_atom2_idx, rdchem.BondType.SINGLE)
                        connected_mol = edit_mol.GetMol()
                        Chem.SanitizeMol(connected_mol)
                        return Chem.MolToSmiles(connected_mol)
                    except Exception as e:
                        continue  # If sanitization fails or invalid connection, try again
                    
    return None  # If no valid connection was found

def show_ring_carbon_numbers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Extract the indices of the ring carbons with exactly one hydrogen atom
    potential_ring_carbons = [atom.GetIdx() for atom in mol.GetAtoms() 
                              if atom.IsInRing() 
                              and atom.GetSymbol() == "C" 
                              and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "H") == 1]
    
    ring_info = mol.GetRingInfo()
    selected_ring_carbons = []
    
    # Further filter the ring carbons based on participation in specific ring sizes
    for ring in ring_info.AtomRings():
        intersect = set(ring).intersection(set(potential_ring_carbons))
        if len(intersect) in [3, 4, 5]:  # If interested in rings of size 3 to 5
            selected_ring_carbons.extend(intersect)
    
    return list(set(selected_ring_carbons))  # Ensure uniqueness and return

def select_carbons_from_different_rings(current_molecule, ring_carbons, n):
    """
    Select up to `n` carbon atoms from different rings within the given molecule.

    Parameters
    ----------
    current_molecule : str
        SMILES representation of the molecule.
    ring_carbons : list
        A list of indices of carbon atoms within rings.
    n : int
        The number of carbon atoms to select, each from a different ring.

    Returns
    -------
    selected_indices : list
        A list of selected carbon atom indices, each from different rings.
    """
    mol = Chem.MolFromSmiles(current_molecule)
    ring_info = mol.GetRingInfo()
    selected_indices = []
    
    for ring in ring_info.AtomRings():
        # Check if this ring contains any of the specified ring carbons
        ring_contains_target_carbon = False
        for idx in ring:
            if idx in ring_carbons and idx not in selected_indices:
                selected_indices.append(idx)
                ring_contains_target_carbon = True
                break  # Break after adding one carbon from the current ring
        
        # Exit the loop early if we've selected the desired number of carbons
        if len(selected_indices) == n:
            break
    
    return selected_indices


def count_atoms(smiles_input):
    """
    Counts the number of each specified element in a single SMILES string or a list of SMILES strings.
    Ensures all expected elements are included in the result dictionary, even if their count is 0.

    Parameters
    ----------
    smiles_input : str or list
        A single SMILES string or a list of SMILES strings representing molecule(s).

    Returns
    -------
    dict
        A dictionary with element symbols as keys and their total counts across all input molecules as values.
        Includes 'C', 'H', 'O', 'N', and 'S'.
    """
    atom_counts = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'S': 0}  # Predefine expected elements with counts of 0

    # Check if the input is a single SMILES string or a list of SMILES strings
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]  # Convert to list for uniform processing
    elif isinstance(smiles_input, list):
        smiles_list = smiles_input
    else:
        print("Unsupported input type. Please provide a SMILES string or a list of SMILES strings.")
        return

    # Traverse the SMILES list
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:  # Check if the molecule object was created successfully
            molecule = Chem.AddHs(molecule)
            for atom in molecule.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_counts:
                    atom_counts[symbol] += 1

    return atom_counts

def balance_c_and_n_atoms(smiles, target_element_counts):
    """
    Modifies a molecule by replacing carbon atoms with nitrogen to meet target element counts,
    while avoiding changes in benzene rings.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    target_element_counts : dict
        Dictionary with target counts for elements, e.g., {'C': x, 'N': y}.

    Returns
    -------
    new_smiles : str
        A new SMILES string of the modified molecule, or the original SMILES if the desired modifications cannot be made.
    """
    mol = Chem.MolFromSmiles(smiles)
    new_mol = Chem.RWMol(mol)  # Create a writable copy of the molecule for modifications.

    # Analyze ring information to avoid replacing carbon atoms in benzene rings
    ssr = Chem.GetSymmSSSR(new_mol)
    benzene_indices = set()
    for ring in ssr:
        if len(ring) == 6:  # Assuming benzene rings are six-membered
            ring_atoms = [new_mol.GetAtomWithIdx(idx) for idx in ring]
            if all(atom.GetSymbol() == 'C' for atom in ring_atoms):
                benzene_indices.update(ring)

    current_c_count = sum(1 for atom in new_mol.GetAtoms() if atom.GetSymbol() == "C")
    target_c_count = target_element_counts.get("C", current_c_count)

    c_to_replace = current_c_count - target_c_count

    if c_to_replace > 0:  # Replace some carbon atoms with nitrogen atoms
        c_indices = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetIdx() not in benzene_indices]
        if c_to_replace > len(c_indices):
            print("Error: Cannot replace more carbon atoms with nitrogen atoms.")
            return Chem.MolToSmiles(new_mol)
        selected_indices = random.sample(c_indices, c_to_replace)
        for idx in selected_indices:
            new_mol.GetAtomWithIdx(idx).SetAtomicNum(7)  # Set atomic number to 7 for nitrogen

    new_smiles = Chem.MolToSmiles(new_mol)
    return new_smiles

def find_atom_indices(smiles, atom_symbol):
    """
    Returns the indices of specified atoms in a molecule. For oxygen atoms,
    it specifically returns those bonded to exactly two carbon atoms.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    atom_symbol : str
        The symbol of the atom to find ('O' for oxygen, 'S' for sulfur).

    Returns
    -------
    list
        A list of indices for the specified atoms under the given conditions.

    """
    molecule = Chem.MolFromSmiles(smiles)
    atom_indices = []

    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == atom_symbol:
            if atom_symbol == 'O':
                # Check if the oxygen is bonded to exactly two carbon atoms
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 2 and all(neighbor.GetSymbol() == 'C' for neighbor in neighbors):
                    atom_indices.append(atom.GetIdx())
            elif atom_symbol == 'S':
                # Directly add the sulfur atom's index, or apply additional checks here if needed
                atom_indices.append(atom.GetIdx())

    return atom_indices

def balance_o_and_s_atoms(smiles, target_element_counts):
    """
    Modifies a molecule by replacing sulfur atoms with oxygen, or vice versa,
    to meet target element counts for oxygen and sulfur.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    target_element_counts : dict
        Dictionary with target counts for oxygen ('O') and sulfur ('S').

    Returns
    -------
    str
        A new SMILES string of the modified molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)
    oxygen_indices = find_atom_indices(smiles, 'O')
    sulfur_indices = find_atom_indices(smiles, 'S')

    current_O_count = len(oxygen_indices)
    current_S_count = len(sulfur_indices)

    periodic_table = Chem.GetPeriodicTable()

    # Replace S with O if current_O_count < target_O_count
    while current_O_count < target_element_counts.get('O', 0) and sulfur_indices:
        idx = random.choice(sulfur_indices)
        molecule.GetAtomWithIdx(idx).SetAtomicNum(periodic_table.GetAtomicNumber('O'))
        sulfur_indices.remove(idx)
        oxygen_indices.append(idx)  # Update indices list
        current_O_count += 1
        current_S_count -= 1

    # Replace O with S if current_S_count < target_S_count
    while current_S_count < target_element_counts.get('S', 0) and oxygen_indices:
        idx = random.choice(oxygen_indices)
        molecule.GetAtomWithIdx(idx).SetAtomicNum(periodic_table.GetAtomicNumber('S'))
        oxygen_indices.remove(idx)
        sulfur_indices.append(idx)  # Update indices list
        current_O_count -= 1
        current_S_count += 1

    return Chem.MolToSmiles(molecule, isomericSmiles=True)

def calculate_unsaturated_carbon_ratio(smiles):
    # 从SMILES字符串创建一个RDKit分子对象
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print("提供的SMILES字符串无效或格式有误。")
        return 0  # 或者根据你的需求返回其他表示错误的值
    
    total_carbons = 0  # 总碳原子数量
    unsaturated_carbons = 0  # 不饱和碳原子数量

    # 遍历分子中的所有原子
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':  # 如果原子是碳原子
            total_carbons += 1
            # 检查碳原子是否不饱和（参与了双键或三键）
            for bond in atom.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:  # 如果发现了双键或三键
                    unsaturated_carbons += 1
                    break  # 不需要检查其他键，一个不饱和键就足够了
    
    # 如果没有碳原子，返回比例0
    if total_carbons == 0:
        return 0
    
    # 计算不饱和碳原子占总碳原子的比例
    ratio = unsaturated_carbons / total_carbons

    print(f"不饱和碳原子数量: {unsaturated_carbons}")

    return ratio

def count_property(smiles_input):
    """
    Prints the counts of specified elements (C, N, H, S, O) in a single SMILES string or a list of SMILES strings.

    Parameters
    - smiles_input: A single SMILES string or a list of SMILES strings representing molecule(s).
    """
    property_counts = {'C_N_ar': 0, 'C_al': 0, 'O_S': 0, 'H': 0}

    # Check if the input is a single SMILES string or a list of SMILES strings
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]  # Convert to list for uniform processing
    elif isinstance(smiles_input, list):
        smiles_list = smiles_input
    else:
        print("Unsupported input type. Please provide a SMILES string or a list of SMILES strings.")
        return

    # Traverse the SMILES list
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_with_hs = Chem.AddHs(mol)

        for atom in mol_with_hs.GetAtoms():
            if atom.GetSymbol() == 'C' or atom.GetSymbol() == 'N':
                if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()):
                    property_counts['C_N_ar'] += 1  # Count as unsaturated C (or N) if any bond is not single
                else:
                    property_counts['C_al'] += 1
            elif atom.GetSymbol() in ['O', 'S']:
                property_counts['O_S'] += 1
            if atom.GetSymbol() == 'H':
                property_counts['H'] += 1

    return property_counts

# # 假设 current_smiles_list 已经定义并包含了SMILES字符串
def drawMolecule(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  # 创建分子时不进行sanitize操作
    img = Draw.MolToImage(mol, size=(600, 600))
    img.show()

def drawMolecules(smiles_list, molsPerRow, maxMols=100):
    """
    Draws a grid image of molecules from a list of SMILES strings.

    Parameters
    - smiles_list: List of SMILES strings representing the molecules to be drawn.
    - molsPerRow: Number of molecules to display per row in the grid image.
    """
    # Convert SMILES strings to RDKit molecule objects and add hydrogens
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    molecules = [AllChem.AddHs(mol) for mol in molecules]

    # Optionally remove hydrogens for a cleaner drawing
    molecules_noH = [Chem.RemoveHs(mol) for mol in molecules]

    # Draw and return the grid image of molecules
    mol_property = [count_property(i) for i in smiles_list]
    img = Draw.MolsToGridImage(molecules_noH, molsPerRow=molsPerRow, subImgSize=(300, 300), maxMols=maxMols, legends=[str(i) for i in mol_property])
    display(img)
    return mol_property

def calculate_mass_percentages(atom_counts):
    """
    Calculates the mass percentages of elements based on their counts in a molecule.

    Parameters
    - atom_counts (dict): A dictionary with element symbols as keys and their counts as values.

    Returns
    - dict: A dictionary with element symbols as keys and their mass percentages as values.
    """
    # 原子质量参考，单位为原子质量单位 (amu)
    atomic_masses = {
        'C': 12.01,
        'H': 1.008,
        'O': 16.00,
        'N': 14.01,
        'S': 32.06
    }
    
    # 计算每个元素的总质量
    masses = {element: count * atomic_masses[element] for element, count in atom_counts.items()}
    
    # 计算总质量
    total_mass = sum(masses.values())
    
    # 计算元素的质量百分比
    mass_percentages = {element: (mass / total_mass) * 100 for element, mass in masses.items()}
    
    return mass_percentages

def ad2daf(C, N, H, S, M, A):
    # 第一步：计算AD基准下的O的百分比
    O = 100 - (C + N + H + S + M + A)
    
    # 计算AD基准下CNHSO的总和
    total_CNHSO_ad = C + N + H + S + O
    print('O_ad=', O)
    
    # 计算daf基准下的CNHSO百分比
    C_daf = C / total_CNHSO_ad * 100
    N_daf = N / total_CNHSO_ad * 100
    H_daf = H / total_CNHSO_ad * 100
    S_daf = S / total_CNHSO_ad * 100
    O_daf = O / total_CNHSO_ad * 100
    
    return C_daf, N_daf, H_daf, S_daf, O_daf

def daf2ad(C_daf, N_daf, H_daf, S_daf, O_daf, M, A):
    # 从daf到ad的转换系数
    conversion_factor = (100 - (M + A)) / 100
    
    # 计算ad基准下的CNHS
    C_ad = C_daf * conversion_factor
    N_ad = N_daf * conversion_factor
    H_ad = H_daf * conversion_factor
    S_ad = S_daf * conversion_factor
    
    # 计算ad基准下的CNHS总和
    total_CNHS_ad = C_ad + N_ad + H_ad + S_ad
    
    # 计算ad基准下的氧百分比
    O_ad = 100 - (total_CNHS_ad + M + A)
    
    return C_ad, N_ad, H_ad, S_ad, O_ad

# # 使用您提供的数据
# C_daf = 83.96
# N_daf = 3.81
# H_daf = 3.31
# S_daf = 0.39
# O_daf = 8.99
# M = 2.51  # 水分
# A = 1.66  # 灰分
				
# # 调用函数
# C_ad, N_ad, H_ad, S_ad, O_ad = ut.daf2ad(C_daf, N_daf, H_daf, S_daf, O_daf, M, A)
# print("ad基准下百分比:", C_ad, N_ad, H_ad, S_ad, O_ad)
# print("CNHSOMA和:", C_ad+N_ad+H_ad+S_ad+O_ad+M+A)


def calculate_MA(C_ad, N_ad, H_ad, S_ad, C_daf, O_daf):
    # 计算ad基准下CNHS的总和
    total_CNHS_ad = C_ad + N_ad + H_ad + S_ad
    
    # 根据比例关系解出O_ad
    O_ad = (C_ad / C_daf) * O_daf
    
    # 计算M和A的总和
    MA_sum = 100 - (total_CNHS_ad + O_ad)
    
    return  MA_sum

# C_ad = 80.92
# N_ad = 3.49   # 氮
# H_ad = 3.03  # 氢
# S_ad = 0.38  # 硫
# C_daf = 83.86
# O_daf = 8.99

# C_ad = 74.37
# N_ad = 1.21   # 氮
# H_ad = 5.687  # 氢
# S_ad = 0.342  # 硫
# C_daf = 82.44
# O_daf = 9.53
# ut.calculate_MA(C_ad, N_ad, H_ad, S_ad, C_daf, O_daf)
