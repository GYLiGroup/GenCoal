from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import AddHs, Draw, AllChem
from rdkit.Chem.Draw import IPythonConsole
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

IPythonConsole.maxMols = 200  # 将最大显示分子数改为200或更大


from joblib import Parallel, delayed

# NMR integral ------------ create json
def calculate_C90_C180(file_path: str) -> Tuple[float, float]:
    """Calculate the integral ratios C90 and C180 from NMR data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing NMR data with columns 'X' and 'Y'.

    Returns
    -------
    Tuple[float, float]
        Ratios C90 and C180.
    """
    # 导入数据
    df = pd.read_csv(file_path)
    
    # 数据清洗
    filtered_df = df[(df['X'] >= 0) & (df['X'] <= 250) & (df['Y'] >= 0)]
    x = filtered_df['X'].values
    y = filtered_df['Y'].values
    
    # 创建线性插值函数
    f = interp1d(x, y, kind='linear')
    x_values = np.linspace(x.min(), x.max(), 1000)
    y_values = f(x_values)

    # 使用Simpson积分法计算总面积
    total_area = simps(y_values, x_values)

    # 计算90到最大x值的面积比
    x_90_max = x_values[x_values >= 90]
    y_90_max = f(x_90_max)
    area_90_max = simps(y_90_max, x_90_max)
    C90 = round(area_90_max / total_area, 3)

    # 计算180到最大x值的面积比
    x_180_max = x_values[x_values >= 180]
    y_180_max = f(x_180_max)
    area_180_max = simps(y_180_max, x_180_max)
    C180 = round(area_180_max / total_area, 3)

    # 绘图
    plt.figure(figsize=(16, 8))
    plt.plot(x_values, y_values, label='Interpolated Curve', color='blue')
    plt.fill_between(x_90_max, y_90_max, color='red', alpha=0.5, label='Area from 90 to Max')
    plt.fill_between(x_180_max, y_180_max, color='blue', alpha=0.5, label='Area from 180 to Max')
    plt.xlabel('Chemical Shift (ppm)', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rc('font', family='Palatino')
    plt.show()

    return C90, C180

# Read JSON data
def read_json(file_name: str) -> dict:
    """Reads a JSON file and returns its content.

    Parameters
    ----------
    file_name : str
        Path to the JSON file.

    Returns
    -------
    dict
        The content of the JSON file.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data


# Calculate moles of each element based on carbon moles and elemental mass ratios
def calculate_element_moles(C_moles: float, ele_ratio: dict) -> dict:
    """Calculate the moles of each element in a compound based on the moles of carbon (C)
    and the mass percentage ratios of other elements relative to carbon.

    Parameters
    ----------
    C_moles : float
        The carbon counts in the compound.
    ele_ratio : dict
        Elemental percentage (e.g., {'C': 80, 'H': 10, 'O': 10}).

    Returns
    -------
    dict
        Number of each element in the compound.
    """
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


# Merge atom counts based on SMILES representation
def merge_count_atoms(smiles: str) -> dict:
    """Calculate the target property of each element of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES of a molecule.

    Returns
    -------
    dict
        A dictionary of atom counts (e.g., {'C_N_ar': 15, 'C_al': 0, 'O_S': 1, 'H': 7}).
    """
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


# Sort SMILES based on elemental compositions
def getPackage(total_smiles_list: list) -> list:
    """Sorts a list of SMILES based on their calculated elemental compositions.
    The sorting criteria prioritize compounds with higher ratios of certain elements to H.

    Parameters
    ----------
    total_smiles_list : list
        A list of SMILES.

    Returns
    -------
    list
        A list of SMILES strings sorted according to their elemental composition ratios.
    """
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


# Calculate the number of carbonyl and hydroxyl groups
def recommended_carbonyl_hydroxyl(C_moles: float, ele_ratio: dict) -> Tuple[int, int]:
    """Calculate the number of carbonyl and hydroxyl groups based on atom counts and compound type.

    Parameters
    ----------
    C_moles : float
        Number of carbon atoms in the compound.
    ele_ratio : dict
        Elemental percentage (e.g., {'C': 80, 'H': 10, 'O': 10}).

    Returns
    -------
    Tuple[int, int]
        The number of carbonyl and hydroxyl groups in the compound.
    """
    atom_num = calculate_element_moles(C_moles, ele_ratio)
    
    total_mass = atom_num['C'] * 12 + atom_num['H'] * 1 + atom_num['O'] * 16 + atom_num['N'] * 14 + atom_num['S'] * 32

    carbonyl = round((0.01 * total_mass) / 16)
    hydroxyl = round((0.02 * total_mass) / 16)

    return carbonyl, hydroxyl


# Adjust the target atom counts for a compound
def getTarget(C90: float, ele_ratio: dict, C_moles: float, carbonyl: int, hydroxyl: int) -> Tuple[dict, dict]:
    """Adjusts the target atom counts for a chemical compound based on specified carbon moles,
    elemental mass ratios, and the percentages of carboxyl and hydroxyl groups.

    Parameters
    ----------
    C90 : float
        The degree of aromaticity.
    ele_ratio : dict
        Elemental mass percentages (e.g., {'C': 80, 'H': 10, 'O': 10}).
    C_moles : float
        The number of carbon atoms.
    carbonyl : int
        The number of carbonyl groups.
    hydroxyl : int
        The number of hydroxyl groups.

    Returns
    -------
    Tuple[dict, dict]
        The calculated chemical formula and the adjusted target atom counts.
    """
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
    

def build_H_nested_dict(sorted_choices: List[Dict[str, int]]) -> Dict[int, Dict[int, int]]:
    """
    Constructs a nested dictionary to store the minimum hydrogen (H) values for combinations 
    of carbon-nitrogen aromatic count (C_N_ar) and oxygen-sulfur count (O_S) from a list of compounds.

    The function iterates over the provided list of compound data (sorted_choices) and for each
    unique combination of C_N_ar (carbon-nitrogen aromatic count) and O_S (oxygen-sulfur count),
    it records the minimum hydrogen (H) value. The result is stored in a nested dictionary format 
    where the first level keys are 'C_N_ar' values, the second level keys are 'O_S' values, and 
    the values are the minimum 'H' values for those combinations.

    Example
    -------
    >>> sorted_choices = [
    >>>     {'C_N_ar': 10, 'O_S': 2, 'H': 12},
    >>>     {'C_N_ar': 10, 'O_S': 2, 'H': 8},
    >>>     {'C_N_ar': 5, 'O_S': 3, 'H': 14}
    >>> ]
    >>> build_H_nested_dict(sorted_choices)
    {10: {2: 8}, 5: {3: 14}}

    Notes
    -----
    - The function assumes that the input list (`sorted_choices`) is pre-sorted by 'C_N_ar' and 'O_S' values.
    - If multiple entries have the same C_N_ar and O_S values, the function will select the minimum H value from those entries.
    - The function is designed to handle situations where not all combinations of C_N_ar and O_S are present, 
      in which case those combinations will not appear in the output dictionary.
    """
    nested_dict = defaultdict(lambda: defaultdict(list))

    # Grouping H values into the nested dictionary by C_N_ar and O_S
    for choice in sorted_choices:
        nested_dict[choice['C_N_ar']][choice['O_S']].append(choice['H'])

    # Finding the minimum H value in each list of H values for a given C_N_ar and O_S combination
    for c_n_ar_key in nested_dict:
        for o_s_key in nested_dict[c_n_ar_key]:
            nested_dict[c_n_ar_key][o_s_key] = min(nested_dict[c_n_ar_key][o_s_key])

    # Convert defaultdict to a standard dictionary for the final output
    return {c_n_ar_key: dict(inner_dict) for c_n_ar_key, inner_dict in nested_dict.items()}

def find_combinations(sorted_choices, target_C_N_ar):
    """
    Finds all unique combinations of 'C_N_ar' values from 'sorted_choices' that sum up to 'target_C_N_ar'.

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
    candidates = list(set(choice['C_N_ar'] for choice in sorted_choices))
    candidates.sort()

    def dfs(start, target, path, res):
        if target == 0:
            res.append(path)
            return
        if target < 0 or start == len(candidates):
            return
        if candidates[start] == 0:
            dfs(start + 1, target, path, res)
            return
        dfs(start, target - candidates[start], path + [candidates[start]], res)
        dfs(start + 1, target, path, res)

    res = []
    dfs(0, target_C_N_ar, [], res)

    cn_list = []
    for combination in res:
        temp_dict = {key: combination.count(key) for key in candidates}
        cn_list.append(temp_dict)
    
    return cn_list


def backtrack_combinations(nested_dict_H: Dict[int, Dict[int, int]], 
                            selection_dic: Dict[int, int], 
                            target_O_S: int, 
                            max_depth: int = 50) -> List[Dict[int, Dict[int, int]]]:
    """
    Finds valid combinations of hydrogen (H) counts that match a specified target for oxygen-sulfur (O_S) count 
    using a backtracking algorithm. This function explores all possible combinations of H and O_S values to 
    find combinations that meet the target O_S sum and the required hydrogen counts specified in `selection_dic`.

    Parameters
    ----------
    nested_dict_H : Dict[int, Dict[int, int]]
        A nested dictionary where the first level keys are hydrogen counts (H) and the second level keys are 
        oxygen-sulfur counts (O_S). The values are the available counts for each hydrogen (H) and oxygen-sulfur (O_S) combination.
    
    selection_dic : Dict[int, int]
        A dictionary where the keys represent hydrogen counts (H) and the values represent the total number of times
        each hydrogen count must be used in the final solution.

    target_O_S : int
        The target sum of oxygen-sulfur (O_S) values that the selected combinations of H and O_S should match.

    max_depth : int, optional
        The maximum depth to which the backtracking algorithm will explore. Default is 30. This helps control the recursion depth
        and avoid excessive computation for large inputs.

    Returns
    -------
    List[Dict[int, Dict[int, int]]]
        A list of dictionaries, where each dictionary represents a valid combination of hydrogen (H) and oxygen-sulfur (O_S)
        values that satisfy the target O_S sum and adhere to the required hydrogen counts as specified in `selection_dic`.

    Example
    -------
    >>> nested_dict_H = {
    >>>     12: {2: 3, 3: 5},
    >>>     8: {2: 4, 3: 2},
    >>> }
    >>> selection_dic = {12: 2, 8: 1}
    >>> target_O_S = 5
    >>> backtrack_combinations(nested_dict_H, selection_dic, target_O_S)
    [{'12': {2: 1, 3: 1}, '8': {2: 1}}]

    Notes
    -----
    - The function employs a depth-first search (DFS) approach to explore potential combinations and uses memoization 
      to avoid recomputing previously explored paths.
    - If a solution exceeds the `max_depth`, the function will stop further exploration.
    - The function returns multiple valid solutions if they exist.
    """
    # Sort H keys in descending order for optimization
    H_keys = sorted(nested_dict_H.keys(), reverse=True)
    solutions = []
    memo = {}

    def convert_to_hashable(current_selection: Dict[int, Dict[int, int]]) -> Tuple:
        """
        Converts the current selection dictionary into a hashable type (tuple) for memoization.

        Parameters
        ----------
        current_selection : Dict[int, Dict[int, int]]
            A dictionary representing the current selection of hydrogen (H) and oxygen-sulfur (O_S) counts. 
            This dictionary is used to track the ongoing selection during backtracking.

        Returns
        -------
        Tuple
            A tuple representation of the current selection dictionary, making it hashable for memoization.
        """
        return tuple((k, tuple(v.items())) for k, v in current_selection.items())

    def backtrack(remaining_O_S: int, H_index: int, current_selection: Dict[int, Dict[int, int]], depth: int) -> List[Dict[int, Dict[int, int]]]:
        """
        Recursive backtracking function to explore combinations of hydrogen (H) and oxygen-sulfur (O_S) values.

        Parameters
        ----------
        remaining_O_S : int
            The remaining oxygen-sulfur sum that needs to be matched during backtracking.

        H_index : int
            The index in the list of hydrogen (H) keys indicating the current hydrogen value being processed.

        current_selection : Dict[int, Dict[int, int]]
            A dictionary representing the current selection of hydrogen (H) and oxygen-sulfur (O_S) counts. 

        depth : int
            The current depth of recursion in the backtracking process. Used to control recursion limit.

        Returns
        -------
        List[Dict[int, Dict[int, int]]]
            A list of valid combinations where the hydrogen counts match the selection criteria, and the O_S sum is met.
        """
        # Base case: Stop if maximum depth is exceeded
        if depth > max_depth:
            return []
        
        # Check memo to avoid re-exploring known paths
        hashable_key = (remaining_O_S, H_index, convert_to_hashable(current_selection))
        if hashable_key in memo:
            return []

        # Successful case: If target O_S is met and all H counts match selection criteria
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
        solutions.extend(backtrack(remaining_O_S, H_index + 1, current_selection, depth + 1))

        # Try adding current H and explore further
        for O_S, count in nested_dict_H[current_H].items():
            if sum(current_selection[current_H].values()) < selection_dic[current_H]:
                current_selection[current_H][O_S] += 1
                solutions.extend(
                    backtrack(remaining_O_S - O_S, H_index, current_selection, depth + 1)
                )
                current_selection[current_H][O_S] -= 1

        memo[hashable_key] = True
        return solutions

    # Initialize the selection with zeros for each possible O_S count under each H key
    initial_selection = {key: {subkey: 0 for subkey in nested_dict_H[key]} for key in nested_dict_H.keys()}
    backtrack(target_O_S, 0, initial_selection, 0)  # Start backtracking with initial conditions

    return solutions

def parallel_backtrack_combinations(nested_dict_H: Dict[int, Dict[int, int]], 
                                    selection_dic: Dict[int, int], 
                                    target_O_S: int, 
                                    max_depth: int = 50, 
                                    n_jobs: int = 40) -> List[Dict[int, Dict[int, int]]]:
    """
    Parallelized version of backtracking to find combinations of hydrogen (H) and oxygen-sulfur (O_S) values 
    that match a specified target O_S sum, using multiple cores for faster computation. This function explores all 
    possible combinations of H and O_S values and finds those that meet the target O_S sum while adhering to the 
    required hydrogen counts specified in `selection_dic`.

    Parameters
    ----------
    nested_dict_H : Dict[int, Dict[int, int]]
        A nested dictionary where the first level keys represent hydrogen counts (H) and the second level keys 
        represent oxygen-sulfur (O_S) values. The values in the dictionary represent the number of times each H-O_S 
        combination can be used.

    selection_dic : Dict[int, int]
        A dictionary where the keys represent hydrogen counts (H) and the values represent the total number of times 
        each hydrogen count must be used in the solution.

    target_O_S : int
        The target sum of oxygen-sulfur (O_S) values that the selected combinations of H and O_S should match.

    max_depth : int, optional
        The maximum depth to which the backtracking algorithm will explore. Default is 50. This parameter is used to control 
        the recursion depth in the backtracking algorithm, helping to avoid excessive computation for larger inputs.

    n_jobs : int, optional
        The number of parallel jobs (cores) to use for processing the subtasks. Default is 40. This controls how many parallel 
        processes the algorithm will use to speed up the backtracking computation.

    Returns
    -------
    List[Dict[int, Dict[int, int]]]
        A list of dictionaries, where each dictionary represents a valid combination of hydrogen (H) and oxygen-sulfur (O_S) 
        values that satisfy the target O_S sum and the required hydrogen counts as specified in `selection_dic`.

    Example
    -------
    >>> nested_dict_H = {
    >>>     12: {2: 3, 3: 5},
    >>>     8: {2: 4, 3: 2},
    >>> }
    >>> selection_dic = {12: 2, 8: 1}
    >>> target_O_S = 5
    >>> parallel_backtrack_combinations(nested_dict_H, selection_dic, target_O_S, max_depth=50, n_jobs=4)
    [{'12': {2: 1, 3: 1}, '8': {2: 1}}]

    Notes
    -----
    - The function divides the work into subtasks for parallelization by distributing the combinations to multiple cores.
    - It uses the `joblib` library for parallel execution, allowing efficient use of multiple CPU cores.
    - If a solution exceeds the `max_depth`, the function will stop further exploration for that branch.
    - The function returns all valid combinations that meet the target O_S sum and hydrogen count criteria.

    """
    # Sort H keys in descending order for optimization
    H_keys = sorted(nested_dict_H.keys(), reverse=True)

    def convert_to_hashable(current_selection: Dict[int, Dict[int, int]]) -> Tuple:
        """
        Converts the current selection dictionary into a hashable type (tuple) for memoization.

        Parameters
        ----------
        current_selection : Dict[int, Dict[int, int]]
            A dictionary representing the current selection of hydrogen (H) and oxygen-sulfur (O_S) counts.

        Returns
        -------
        Tuple
            A tuple representation of the current selection dictionary, making it hashable for memoization.
        """
        return tuple((k, tuple(v.items())) for k, v in current_selection.items())

    def backtrack(remaining_O_S: int, H_index: int, current_selection: Dict[int, Dict[int, int]], 
                  depth: int, memo: Dict) -> List[Dict[int, Dict[int, int]]]:
        """
        Recursive backtracking function to explore combinations of hydrogen (H) and oxygen-sulfur (O_S) values.

        Parameters
        ----------
        remaining_O_S : int
            The remaining oxygen-sulfur sum that needs to be matched during backtracking.

        H_index : int
            The index in the list of hydrogen (H) keys indicating the current hydrogen value being processed.

        current_selection : Dict[int, Dict[int, int]]
            A dictionary representing the current selection of hydrogen (H) and oxygen-sulfur (O_S) counts.

        depth : int
            The current depth of recursion in the backtracking process. Used to control recursion limit.

        memo : Dict
            A dictionary used for memoization to avoid redundant calculations.

        Returns
        -------
        List[Dict[int, Dict[int, int]]]
            A list of valid combinations where the hydrogen counts match the selection criteria, and the O_S sum is met.
        """
        # Base case: Stop if maximum depth is exceeded
        if depth > max_depth:
            return []
        
        # Check memo to avoid re-exploring known paths
        hashable_key = (remaining_O_S, H_index, convert_to_hashable(current_selection))
        if hashable_key in memo:
            return []

        # Successful case: If target O_S is met and all H counts match selection criteria
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

    def generate_subtasks(nested_dict_H: Dict[int, Dict[int, int]]) -> List[Tuple[int, int]]:
        """
        Generate finer-grained subtasks for parallelization by iterating through H and O_S combinations.

        Parameters
        ----------
        nested_dict_H : Dict[int, Dict[int, int]]
            A nested dictionary where the first level keys represent hydrogen counts (H) and the second level keys
            represent oxygen-sulfur counts (O_S). The values represent the number of available combinations for each.

        Returns
        -------
        List[Tuple[int, int]]
            A list of subtasks, where each subtask is a tuple consisting of a hydrogen count (H) and an oxygen-sulfur
            count (O_S).
        """
        subtasks = []
        for H_key, sub_dict in nested_dict_H.items():
            for O_S_key in sub_dict.keys():
                subtasks.append((H_key, O_S_key))
        return subtasks

    def backtrack_for_subtask(H_key: int, O_S_key: int) -> List[Dict[int, Dict[int, int]]]:
        """
        Perform backtracking for a specific H and O_S combination.

        Parameters
        ----------
        H_key : int
            The hydrogen count (H) for this subtask.

        O_S_key : int
            The oxygen-sulfur count (O_S) for this subtask.

        Returns
        -------
        List[Dict[int, Dict[int, int]]]
            A list of valid combinations for the given H and O_S combination.
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


def find_matching_indices(selection: Dict[int, Dict[int, int]], 
                          sorted_choices: List[Dict[str, int]]) -> List[int]:
    """
    Finds indices in 'sorted_choices' that match the criteria specified in 'selection',
    considering constraints on repetition and randomly selecting from matching indices 
    while adhering to the repetition limits.

    This function takes a nested dictionary 'selection' that specifies how many times each 
    combination of 'C_N_ar' and 'O_S' should be selected, then finds the corresponding 
    indices from 'sorted_choices' that meet these criteria while enforcing limits on the 
    number of times each index can be chosen.

    Parameters
    ----------
    selection : dict
        A nested dictionary where the first level keys represent 'C_N_ar' values, and the second 
        level keys represent 'O_S' values. The values are integers indicating how many times that 
        specific combination of 'C_N_ar' and 'O_S' should be selected.

    sorted_choices : list of dicts
        A list where each dictionary represents a possible choice. Each dictionary contains keys such 
        as 'C_N_ar' and 'O_S' among others. These are the possible combinations from which the function 
        selects the matching indices according to the specified criteria in 'selection'.

    Returns
    -------
    indices : list of int
        A list of indices from 'sorted_choices' that match the 'selection' criteria. Each index corresponds 
        to an entry in 'sorted_choices' that fits one of the combinations specified in 'selection'. 
        The indices are chosen while respecting the maximum repetition allowed for each index.

    Example
    -------
    >>> selection = {2: {3: 2}, 4: {5: 1}}
    >>> sorted_choices = [{'C_N_ar': 2, 'O_S': 3}, {'C_N_ar': 2, 'O_S': 3}, {'C_N_ar': 4, 'O_S': 5}]
    >>> find_matching_indices(selection, sorted_choices)
    [0, 1, 2]
    
    Notes
    -----
    - The function ensures that each index is selected within the limit specified by 'selection'.
    - Random selection is used from matching indices, and the selection is done uniformly with respect to 
      the repetition constraints.
    - If there are more requested selections than matching options, the available indices will be selected 
      multiple times while adhering to the specified repeat limits.
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

def generate_candidate_smiles(min_H_selection: Dict[int, Dict[int, int]], 
                               sorted_choices: List[Dict[str, int]], 
                               sorted_smiles_list: List[str], 
                               target_C_al: int) -> Tuple[List[str], str]:
    """
    Generates a list of candidate SMILES strings based on a selection criteria, 
    and calculates the predicted chemical formula for these candidates, adjusting 
    for a target count of aliphatic carbon (C_al).

    This function selects candidate SMILES strings from a given list based on the hydrogen 
    selection criteria provided in `min_H_selection`. After selecting the candidates, 
    it calculates the total number of aliphatic carbon atoms (C_al) and adjusts the selection 
    to meet the target number of aliphatic carbons specified by `target_C_al`. 

    Parameters
    ----------
    min_H_selection : dict
        A nested dictionary specifying the minimum selection criteria for hydrogen counts in 
        the candidates. The keys represent the hydrogen groupings, and the values indicate 
        the number of selections for each combination.

    sorted_choices : list of dicts
        A list of dictionaries where each dictionary contains properties like 'C_N_ar', 'C_al', 
        'O_S', and 'H' counts corresponding to possible chemical choices.

    sorted_smiles_list : list of str
        A list of SMILES strings corresponding to the chemical choices in `sorted_choices`.

    target_C_al : int
        The target total number of aliphatic carbon atoms (C_al) that the final set of selected 
        candidates should contain.

    Returns
    -------
    tuple
        - A list of selected candidate SMILES strings that meet the selection criteria.
        - A string representing the predicted chemical formula of the combined candidates, 
          adjusting for any additional carbon atoms to meet the target C_al count.

    Example
    -------
    >>> min_H_selection = {2: {3: 2}, 4: {5: 1}}
    >>> sorted_choices = [{'C_N_ar': 2, 'C_al': 3, 'O_S': 2, 'H': 5}, {'C_N_ar': 2, 'C_al': 4, 'O_S': 1, 'H': 4}]
    >>> sorted_smiles_list = ['CC(C)C', 'CCO']
    >>> target_C_al = 8
    >>> generate_candidate_smiles(min_H_selection, sorted_choices, sorted_smiles_list, target_C_al)
    (['CC(C)C', 'CCO', 'C', 'C'], 'C4H8O2')

    Notes
    -----
    - The function selects candidates based on hydrogen counts and the corresponding 
      SMILES strings.
    - If the total C_al value from selected candidates is less than the target, extra 
      carbon atoms are added (represented as 'C') to meet the required target.
    - The final chemical formula is calculated based on the selected candidates and the 
      added carbon atoms.
    """
    # Find matching indices based on the minimum hydrogen selection criteria
    indices = find_matching_indices(min_H_selection, sorted_choices)
    
    # Generate a list of selected candidate SMILES strings based on the indices
    candidate_smiles_list = [sorted_smiles_list[i] for i in indices]
    
    # Generate legends (string representations) of the selected choices
    selected_legends = [str(sorted_choices[i]) for i in indices]

    # Extract 'C_al' values from the selected choices' legends
    C_al_values = [ast.literal_eval(legend)['C_al'] for legend in selected_legends]
    total_C_al = sum(C_al_values)  # Calculate the total C_al count
    additional_C_al = target_C_al - total_C_al  # Calculate how many extra carbon atoms are needed
    added_value = max(additional_C_al, 0)  # Ensure no negative values for added carbon atoms

    # Add extra carbon atoms if necessary to meet the target C_al
    for _ in range(added_value):
        candidate_smiles_list.append('C')

    # Count the atoms in the selected candidates and the additional carbon atoms
    element_counts = count_atoms(candidate_smiles_list)

    return candidate_smiles_list, element_counts

def find_min_H_selection(self, all_final_selections: list) -> dict:
    """
    Finds the selection from all possible combinations that results in the minimum total hydrogen (H) count.

    This function evaluates a list of possible selections (`all_final_selections`) and calculates the total 
    hydrogen (H) count for each selection. It then returns the selection with the minimum hydrogen count. 
    The total H count for a selection is computed by iterating through each hydrogen value and its respective 
    counts in the selection, using the data from `self.nested_dict_H` to calculate the contribution of each 
    combination.

    Parameters
    ----------
    all_final_selections : list of dict
        A list of dictionaries where each dictionary represents a potential selection of combinations. 
        Each dictionary contains keys representing hydrogen (H) values and their associated oxygen-sulfur (O_S) values, 
        with counts indicating how many times a specific combination is chosen.

    Returns
    -------
    dict
        The selection (a dictionary) from `all_final_selections` that has the minimum total hydrogen count. 
        If there are multiple selections with the same total hydrogen count, the first one encountered will be returned.

    Example
    -------
    >>> all_final_selections = [
    >>>     {2: {3: 1, 4: 2}, 4: {5: 1}},
    >>>     {2: {3: 3}, 4: {5: 2}},
    >>> ]
    >>> find_min_H_selection(all_final_selections)
    {2: {3: 1, 4: 2}, 4: {5: 1}}

    Notes
    -----
    - The function calculates the total hydrogen count by iterating through each hydrogen value (H) 
      and its respective oxygen-sulfur (O_S) combinations, multiplying the counts by the corresponding 
      values from `self.nested_dict_H`.
    - The selection with the smallest total H count is returned. This can be useful for minimizing resource 
      usage or optimizing a process that relies on hydrogen consumption.

    """
    # Initialize variables to track the minimum hydrogen count and the corresponding selection
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


def show_atom_number(mol: Chem.Mol, label: str) -> Chem.Mol:
    """
    Label each atom in the given molecule with its index.

    This function modifies the input RDKit molecule by adding a property to each atom 
    that stores its index. This index can later be accessed using the provided label.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object whose atoms are to be labeled. This object is modified in place.
    label : str
        The label under which the atom index will be stored. This label can then be used to access the atom index from the atom properties.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The RDKit molecule object with atoms labeled with their indices. Note that the molecule is modified in place, 
        so the returned object is the same as the input object.
    
    Example
    -------
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> labeled_mol = show_atom_number(mol, 'atom_index')
    >>> for atom in labeled_mol.GetAtoms():
    >>>     print(atom.GetProp('atom_index'))
    0
    1
    2

    Notes
    -----
    - The atom index is stored as a string property under the specified `label`.
    - This function is useful for identifying and tracking individual atoms in a molecule, especially when performing atom-level analysis.
    """
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol


def find_required_aldehyde_carbons(smiles: str) -> list:
    """
    Identifies carbon atoms in a molecule that are part of an aldehyde group. 
    An aldehyde carbon is defined as a carbon atom having exactly three neighbors: 
    one carbon (C), one hydrogen (H), and one oxygen (O).

    This function processes a SMILES string to identify the aldehyde group carbon atoms 
    by checking their neighbors and ensures that the carbon atom meets the criteria for being part of an aldehyde group.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that are part of an aldehyde group within the molecule. 
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.

    Example
    -------
    >>> find_required_aldehyde_carbons('C=O')
    [0]

    Notes
    -----
    - The function creates an RDKit molecule object from the provided SMILES string and adds hydrogens explicitly to the molecule for better atom detection.
    - The carbon atoms are considered part of an aldehyde if they are bonded to one hydrogen, one carbon, and one oxygen atom.
    - This can be useful in synthetic chemistry or molecular analysis where aldehyde functionalities need to be identified.
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


def find_beta_carbons(smiles: str) -> list:
    """
    Identifies pairs of aldehyde (carbonyl) carbon atoms and their beta carbon atoms in a molecule.

    This function finds aldehyde carbon atoms (part of a carbonyl group) and identifies the beta carbon atoms 
    that are bonded to them. Beta carbon atoms are those directly bonded to an alpha carbon, which is 
    directly bonded to the aldehyde carbon. The function returns a list of tuples where each tuple contains 
    the index of an aldehyde carbon atom and the index of a corresponding beta carbon atom.

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

    Example
    -------
    >>> find_beta_carbons('CC(=O)C')
    [(1, 0)]

    Notes
    -----
    - The function identifies aldehyde (carbonyl) carbon atoms and traces their alpha and beta carbons.
    - The beta carbon is selected only if it is bonded to at least one hydrogen atom, ensuring that it is a valid beta carbon.
    - This function is useful in organic chemistry when looking for specific structural motifs around aldehyde groups.
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

def connect_rings_C3(smiles1: str) -> str or None:
    """
    Connects a given molecule with a propane molecule at identified beta carbon positions.

    This function takes an initial molecule represented by a SMILES string and connects it to a propane molecule
    ('CCC') at the beta carbon positions identified in the initial molecule. If suitable beta carbon atoms 
    (those that are part of aldehyde groups) are found, it adds a propane unit at those positions and returns 
    the modified molecule as a new SMILES string. If no suitable positions are found, it returns None.

    Parameters
    ----------
    smiles1 : str
        A SMILES string representing the initial molecule to which propane will be connected.

    Returns
    -------
    str or None
        A SMILES string representing the modified molecule after connection with propane. 
        If no suitable carbon atoms are found for the connection, returns None.

    Example
    -------
    >>> connect_rings_C3('CC(=O)C')
    'CC(=O)CCC'

    Notes
    -----
    - This function uses the RDKit library to manipulate molecular structures.
    - The propane molecule is added at the beta carbon positions found in the input molecule.
    - If no suitable aldehyde carbon atoms (those that can be connected to propane) are found, the function will print 
      a message and return None.
    - The function assumes that the propane molecule ('CCC') can be directly connected to the identified positions 
      without additional steric or electronic considerations.
    """
    smiles2 = 'CCC'
    
    # Convert input SMILES strings to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # Find beta carbon positions in the initial molecule
    carbons1_list = find_beta_carbons(smiles1)

    # If no suitable beta carbon atoms are found, return None
    if not carbons1_list:
        print("No suitable aldehyde carbon atom found in the molecule")
        return None

    # For each identified pair of aldehyde and beta carbon atoms, combine molecules
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

        # Update mol1 to the modified molecule
        mol1 = edit_combined.GetMol()
        Chem.SanitizeMol(mol1)

    # Return the SMILES string for the modified molecule
    return Chem.MolToSmiles(mol1)

def update_smiles_lists(smiles_list1: list, smiles_list2: list) -> tuple or None:
    """
    Processes two lists of SMILES strings: connects propane molecules to the first list's molecules if specific criteria are met, 
    and then adjusts the second list by removing elements to compensate for the carbons added to the first list.

    This function iterates through the first list of SMILES (`smiles_list1`) and connects a propane molecule ('CCC') 
    to the molecules that meet specific criteria (those containing aldehyde carbon atoms). It then calculates the 
    difference in the number of carbon atoms between the original and modified molecules in the first list. 
    Based on this difference, it adjusts the second list (`smiles_list2`) by removing the necessary number of elements 
    to compensate for the added carbons.

    Parameters
    ----------
    smiles_list1 : list of str
        The first list of SMILES strings to be potentially modified by connecting propane molecules. 
        Each molecule in this list is checked for the presence of aldehyde carbon atoms and modified if necessary.
    smiles_list2 : list of str
        The second list of SMILES strings, from which elements will be removed to compensate for the carbons added to the first list.

    Returns
    -------
    tuple of lists or None
        A tuple containing two lists:
        - The modified first list with propane molecules connected as applicable.
        - The adjusted second list with elements removed to compensate for added carbons.
        
        If there are not enough elements in the second list to compensate for the added carbons, returns None.

    Example
    -------
    >>> smiles_list1 = ['CC(=O)C', 'CCC']
    >>> smiles_list2 = ['C', 'C', 'C', 'C', 'C', 'C']
    >>> update_smiles_lists(smiles_list1, smiles_list2)
    (['CC(=O)CCC', 'CCC'], ['C', 'C', 'C', 'C'])

    Notes
    -----
    - The function uses the `connect_rings_C3` and `find_required_aldehyde_carbons` functions to identify which molecules 
      in the first list require modification.
    - If the second list does not have enough elements to accommodate the added carbons, the function prints an error message 
      and returns `None`.
    - The function assumes that elements in the second list are simple carbon atoms represented by the SMILES string `'C'`.
    """
    new_smiles_list1 = []
    carbon_diff = 0
    
    # Iterate through the first list and process each molecule
    for smiles in smiles_list1:
        if find_required_aldehyde_carbons(smiles):
            new_smiles = connect_rings_C3(smiles)
        else:
            new_smiles = smiles
        new_smiles_list1.append(new_smiles)

        # Calculate the change in the number of carbon atoms
        old_carbons = smiles.count('C')
        new_carbons = new_smiles.count('C')
        carbon_diff += new_carbons - old_carbons

    # Check if there are enough elements in the second list to compensate for the added carbons
    if carbon_diff > len(smiles_list2):
        print(f"Not enough carbons in the second list to compensate the increase in the first list. Needed: {carbon_diff}, Available: {len(smiles_list2)}")
        return None

    # Adjust the second list by removing elements
    new_smiles_list2 = smiles_list2[carbon_diff:]
    
    return new_smiles_list1, new_smiles_list2


def find_required_carbons(smiles: str) -> list:
    """
    Identifies carbon atoms in a molecule that are connected to exactly two other carbon (or nitrogen) atoms and one hydrogen atom.
    
    This function analyzes the given SMILES string and identifies carbon atoms that satisfy the following conditions:
    - The carbon atom is connected to exactly two other carbon (or nitrogen) atoms.
    - The carbon atom is also connected to exactly one hydrogen atom.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure to be analyzed. This string should be valid and represent a molecule in which 
        carbon atoms meet the specified connectivity criteria.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the specified connectivity criteria within the molecule. 
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.

    Example
    -------
    >>> find_required_carbons('CC(N)C')
    [0]

    Notes
    -----
    - The function uses RDKit to parse the SMILES string and then explicitly adds hydrogens to the molecule for proper analysis.
    - The function checks each carbon atom in the molecule for connectivity with exactly two other carbons (or nitrogen atoms) 
      and exactly one hydrogen atom. If these conditions are met, the atom's index is added to the result list.
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

def find_required_carbons(smiles: str) -> list:
    """
    Identifies carbon atoms in a molecule that are connected to exactly two other carbon (or nitrogen) atoms and one hydrogen atom.
    
    This function analyzes the given SMILES string and identifies carbon atoms that satisfy the following conditions:
    - The carbon atom is connected to exactly two other carbon (or nitrogen) atoms.
    - The carbon atom is also connected to exactly one hydrogen atom.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure to be analyzed. This string should be valid and represent a molecule in which 
        carbon atoms meet the specified connectivity criteria.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the specified connectivity criteria within the molecule. 
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.

    Example
    -------
    >>> find_required_carbons('CC(N)C')
    [0]

    Notes
    -----
    - The function uses RDKit to parse the SMILES string and then explicitly adds hydrogens to the molecule for proper analysis.
    - The function checks each carbon atom in the molecule for connectivity with exactly two other carbons (or nitrogen atoms) 
      and exactly one hydrogen atom. If these conditions are met, the atom's index is added to the result list.
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

def _connect_ring_C(mol1: Chem.Mol, mol2: Chem.Mol, chosen_pair1: tuple, chosen_pair2: tuple) -> Chem.Mol:
    """
    Connects two molecules using two methane molecules to simulate the addition of methyl groups at specified carbon atoms.

    This function modifies the input molecules `mol1` and `mol2` by adding two methyl groups (from two methane molecules) at
    the specified carbon atom positions. The carbon atoms are connected according to the indices in `chosen_pair1` for 
    `mol1` and `chosen_pair2` for `mol2`.

    Parameters
    ----------
    mol1 : Chem.Mol
        The first molecule object to be connected. This should be an RDKit molecule.
    mol2 : Chem.Mol
        The second molecule object to be connected. This should be an RDKit molecule.
    chosen_pair1 : tuple
        A tuple of two integers representing the indices of carbon atoms in `mol1` where the methane groups will be attached.
    chosen_pair2 : tuple
        A tuple of two integers representing the indices of carbon atoms in `mol2` where the methane groups will be attached.

    Returns
    -------
    Chem.Mol
        A new RDKit Mol object representing the connected molecule structure. This molecule contains the two original 
        molecules with the addition of two methane groups at the specified positions.

    Example
    -------
    >>> mol1 = Chem.MolFromSmiles('CCO')
    >>> mol2 = Chem.MolFromSmiles('CC')
    >>> chosen_pair1 = (0, 1)
    >>> chosen_pair2 = (0, 1)
    >>> new_mol = _connect_ring_C(mol1, mol2, chosen_pair1, chosen_pair2)
    
    Notes
    -----
    - This function uses RDKit's `CombineMols` to combine the two molecules and add the methane groups.
    - The `chosen_pair1` and `chosen_pair2` should correspond to valid carbon atom indices in `mol1` and `mol2`.
    - The new molecule is sanitized after the connection to ensure validity.
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

def _find_index_pairs(carbons: list, mol: Chem.Mol) -> list:
    """
    Finds index pairs of carbon atoms within a molecule that are suitable for connection.

    This function takes a list of carbon atom indices and identifies pairs of carbon atoms 
    within the molecule that are neighbors. The index pairs are returned in a list of tuples, 
    where each tuple represents a unique pair of carbon atoms.

    Parameters
    ----------
    carbons : list of int
        A list containing indices of carbon atoms within the molecule that are suitable for connection. 
        These indices should correspond to atoms in the `mol` molecule.
    mol : Chem.Mol
        The RDKit Mol object representing the molecule in which the carbon atoms are located. 

    Returns
    -------
    list of tuples
        A list of tuples where each tuple contains two integers. Each integer represents an index of a 
        carbon atom that is connected to another carbon atom in the molecule. The indices are zero-based and 
        correspond to the order in which atoms appear in the RDKit molecule object. 

    Example
    -------
    >>> carbons = [0, 2, 4]
    >>> mol = Chem.MolFromSmiles('CCOCC')
    >>> index_pairs = _find_index_pairs(carbons, mol)
    >>> print(index_pairs)
    [(0, 2), (2, 4)]
    
    Notes
    -----
    - This function ensures that each pair of carbon atoms is returned only once, 
      i.e., it avoids returning the same pair in reverse order (e.g., (0, 2) and (2, 0)).
    - The function assumes that the list `carbons` only contains valid indices that refer to carbon atoms 
      within the provided molecule.
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

def connect_rings(molecules_tuple_list: list) -> str:
    """
    Connects two rings from a list of molecule tuples by adding two methane groups between them.

    This function takes a list of molecule tuples, where each tuple contains a SMILES string 
    representing a molecule and a list of indices of carbon atoms that can be used for connection.
    It selects one pair of carbon atoms from each molecule and connects them by adding two methane 
    groups at the chosen positions. The final connected molecule is then returned as a SMILES string.

    Parameters
    ----------
    molecules_tuple_list : list of tuples
        A list where each tuple contains:
        - A SMILES string representing a molecule.
        - A list of integers, where each integer is an index of a carbon atom that is available for connection.

    Returns
    -------
    str
        A new SMILES string representing the connected molecule structure. If the input list is invalid (less than 2 molecules),
        an error message is returned instead.

    Example
    -------
    >>> molecules = [('C1CCCC1', [0, 1, 2, 3, 4]), ('C1CCCCC1', [0, 1, 2, 3, 4, 5])]
    >>> connect_rings(molecules)
    'CC1CC(C)C(C)C1'

    Notes
    -----
    - The function selects a pair of carbon atoms from each molecule and connects them by adding two methane groups between them.
    - The function relies on the `_find_index_pairs` and `_connect_ring_C` functions for finding suitable carbon pairs and connecting the molecules.
    - If there is less than two molecules in the input list, an appropriate message is returned.
    """
    if len(molecules_tuple_list) < 2:
        # If there is only one tuple in the list, return the SMILES string of that tuple
        if len(molecules_tuple_list) == 1:
            return molecules_tuple_list[0][0]
        # If the list is empty or contains fewer than 2 molecules, return an error message
        print(f"Invalid input: {molecules_tuple_list}")
        return "Invalid input. The list must contain at least two molecules."

    smiles1, carbons1 = molecules_tuple_list[0]
    smiles2, carbons2 = molecules_tuple_list[1]

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    index_pairs_carbons1 = _find_index_pairs(carbons1, mol1)
    index_pairs_carbons2 = _find_index_pairs(carbons2, mol2)

    chosen_pair1 = random.choice(index_pairs_carbons1)
    chosen_pair2 = random.choice(index_pairs_carbons2)

    # Call the helper function to connect the molecules with methane
    final_connected_mol = _connect_ring_C(mol1, mol2, chosen_pair1, chosen_pair2)

    final_connected_smiles = Chem.MolToSmiles(final_connected_mol)
    return final_connected_smiles

def connect_rings(molecules_tuple_list: list) -> str:
    """
    Connects two rings from a list of molecule tuples by adding two methane groups between them.

    This function takes a list of molecule tuples, where each tuple contains a SMILES string 
    representing a molecule and a list of indices of carbon atoms that can be used for connection.
    It selects one pair of carbon atoms from each molecule and connects them by adding two methane 
    groups at the chosen positions. The final connected molecule is then returned as a SMILES string.

    Parameters
    ----------
    molecules_tuple_list : list of tuples
        A list where each tuple contains:
        - A SMILES string representing a molecule.
        - A list of integers, where each integer is an index of a carbon atom that is available for connection.

    Returns
    -------
    str
        A new SMILES string representing the connected molecule structure. If the input list is invalid (less than 2 molecules),
        an error message is returned instead.

    Example
    -------
    >>> molecules = [('C1CCCC1', [0, 1, 2, 3, 4]), ('C1CCCCC1', [0, 1, 2, 3, 4, 5])]
    >>> connect_rings(molecules)
    'CC1CC(C)C(C)C1'

    Notes
    -----
    - The function selects a pair of carbon atoms from each molecule and connects them by adding two methane groups between them.
    - The function relies on the `_find_index_pairs` and `_connect_ring_C` functions for finding suitable carbon pairs and connecting the molecules.
    - If there is less than two molecules in the input list, an appropriate message is returned.
    """
    if len(molecules_tuple_list) < 2:
        # If there is only one tuple in the list, return the SMILES string of that tuple
        if len(molecules_tuple_list) == 1:
            return molecules_tuple_list[0][0]
        # If the list is empty or contains fewer than 2 molecules, return an error message
        print(f"Invalid input: {molecules_tuple_list}")
        return "Invalid input. The list must contain at least two molecules."

    smiles1, carbons1 = molecules_tuple_list[0]
    smiles2, carbons2 = molecules_tuple_list[1]

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    index_pairs_carbons1 = _find_index_pairs(carbons1, mol1)
    index_pairs_carbons2 = _find_index_pairs(carbons2, mol2)

    chosen_pair1 = random.choice(index_pairs_carbons1)
    chosen_pair2 = random.choice(index_pairs_carbons2)

    # Call the helper function to connect the molecules with methane
    final_connected_mol = _connect_ring_C(mol1, mol2, chosen_pair1, chosen_pair2)

    final_connected_smiles = Chem.MolToSmiles(final_connected_mol)
    return final_connected_smiles

def count_hydroxy_oxygen(smiles_list: List[str]) -> int:
    """
    计算一个 SMILES 字符串列表中所有分子中的羟基氧（OH）原子的总数。

    :param smiles_list: SMILES 字符串的列表
    :return: 羟基氧（OH）原子的总数
    """
    total_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)  # 从 SMILES 创建分子对象
        mol_with_hs = AddHs(mol)  # 为分子添加氢原子

        hydroxy_oxygen_indices = []
        for atom in mol_with_hs.GetAtoms():
            # 查找氧原子或硫原子
            if atom.GetSymbol() in ['O', 'S']:
                neighbors = atom.GetNeighbors()
                has_hydrogen = False
                # 检查该原子是否与氢原子相连
                for neighbor in neighbors:
                    if neighbor.GetSymbol() == 'H':
                        has_hydrogen = True
                        break
                if has_hydrogen:
                    hydroxy_oxygen_indices.append(atom.GetIdx())  # 记录原子索引
        total_count += len(hydroxy_oxygen_indices)  # 累加数量
    return total_count

def count_ketone_carbons(smiles_list: List[str]) -> int:
    """
    Counts the total number of ketone carbon atoms in a list of molecules represented by SMILES strings.

    A ketone carbon atom is defined as a carbon atom that is double-bonded to an oxygen atom and is not part of an aldehyde or carboxyl group.

    This function iterates over the provided list of SMILES strings, converts each molecule to an RDKit molecule object, 
    adds explicit hydrogen atoms to the molecule structure (if not already present), and identifies carbon atoms that are 
    part of ketone functional groups. The total count of such carbon atoms is returned.

    Parameters
    ----------
    smiles_list : list of str
        A list containing SMILES strings of the molecules to analyze. Each string represents a molecule.

    Returns
    -------
    total_count : int
        The total count of ketone carbon atoms across all molecules in the given list. 
        A ketone carbon is a carbon atom double-bonded to an oxygen atom.

    Example
    -------
    >>> smiles = ['CC(=O)C', 'C1CC(=O)C1', 'CC(C(=O)C)C']
    >>> count_ketone_carbons(smiles)
    3

    Notes
    -----
    - The function uses RDKit to parse the SMILES strings and explicitly adds hydrogen atoms to the molecule to ensure 
      the bonds are correctly interpreted.
    - Ketone carbons are identified based on the presence of a double bond between a carbon atom and an oxygen atom.
    - This function does not distinguish between ketones and other carbonyl compounds (such as aldehydes or carboxylic acids) 
      because it looks for a carbon double-bonded to an oxygen atom without further specificity.
    - This function is useful for counting ketone functional groups in molecules.

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

def find_C4_carbons(smiles: str) -> list:
    """
    Identifies carbon atoms in a molecule that are bonded to exactly two carbon or nitrogen atoms 
    and have one or two hydrogen atoms.

    The function iterates over the atoms in the molecule, checks if a carbon atom meets the bonding 
    criteria (bonded to two carbon or nitrogen atoms and one or two hydrogen atoms), and returns 
    a list of the indices of such carbon atoms.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the molecular structure to be analyzed. 
        The string encodes the connectivity of atoms and bonds within the molecule.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the specified bonding criteria 
        within the molecule. Atom indices are zero-based, corresponding to the order in which atoms 
        appear in the RDKit molecule object.

    Example
    -------
    >>> smiles = 'CC(C)C'
    >>> find_C4_carbons(smiles)
    [1, 2]

    Notes
    -----
    - The function uses RDKit to convert the SMILES string into a molecule object and explicitly 
      adds hydrogen atoms to ensure accurate bond interpretation.
    - The criteria for selecting carbon atoms are:
        1. The atom must be bonded to exactly two carbon or nitrogen atoms.
        2. The atom must have one or two hydrogen atoms bonded to it.
    - This function is useful for identifying specific carbon atoms that fit certain structural 
      patterns, such as those in substituted hydrocarbons or nitrogen-containing compounds.

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

def connect_rings_C4(smiles1: str) -> str:
    """
    Connects a given molecule with a butane molecule at specified carbon atom positions.

    This function finds carbon atoms in the input molecule (represented by its SMILES string) 
    that meet the criteria for connecting to a butane molecule at its terminal carbons. 
    If suitable carbon atoms are found, the butane molecule is connected, and the resulting 
    molecule is returned as a SMILES string. If no suitable carbon atoms are found, 
    it returns None.

    Parameters
    ----------
    smiles1 : str
        A SMILES string representing the initial molecule to which butane will be connected. 
        The function will attempt to identify specific carbon atoms in this molecule for the connection.

    Returns
    -------
    str or None
        A SMILES string representing the modified molecule after connecting with butane. 
        If no suitable carbon atoms are found for the connection, returns None.

    Example
    -------
    >>> smiles1 = 'CC(C)C'
    >>> connect_rings_C4(smiles1)
    'CC(C)CCCC'  # Example output, depending on the initial molecule

    Notes
    -----
    - The function uses the `find_C4_carbons` function to identify potential carbon atoms in the 
      input molecule that are suitable for bonding with butane. These atoms must meet specific 
      bonding criteria (bonded to two carbons or nitrogens and one or two hydrogens).
    - The function attempts to connect the molecule to butane (a four-carbon chain, SMILES: 'CCCC').
    - The connection is made by bonding the terminal carbons of butane to two carbon atoms from 
      the original molecule.
    - If no suitable pair of carbon atoms for bonding is found, the function prints a message 
      and returns None.
    - The function utilizes RDKit to perform molecule manipulation and bonding.
    """
    smiles2 = 'CCCC'  # SMILES representation of butane
    carbons2 = [0, 3]  # Butane's terminal carbon indices (illustrative, not used directly)

    # Find suitable C4 carbons in the first molecule
    carbons1 = find_C4_carbons(smiles1)

    if not carbons1:
        print("No suitable carbon atom found in the molecule")
        return None

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Identify adjacent carbon pairs in the first molecule for connection
    index_pairs_carbons1 = []
    for i in range(len(carbons1)):
        atom1 = mol1.GetAtomWithIdx(carbons1[i])
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() in carbons1:
                if carbons1[i] < neighbor.GetIdx():
                    index_pairs_carbons1.append((carbons1[i], neighbor.GetIdx()))

    if not index_pairs_carbons1:
        print(f"No suitable pair of carbon atoms found in the molecule: {smiles1}")
        return None

    # Randomly choose a pair of carbon atoms to connect with butane
    chosen_pair1 = random.choice(index_pairs_carbons1)

    # Combine the two molecules: the initial molecule and butane
    combined = Chem.CombineMols(mol1, mol2)
    edit_combined = Chem.EditableMol(combined)

    # Recalculate the atom indices in the combined molecule
    index1, index2 = chosen_pair1  # Indices for carbons in mol1
    index3 = len(mol1.GetAtoms())  # Index for the first atom of mol2 (butane)
    index4 = len(mol1.GetAtoms()) + 3  # Index for the last atom of mol2 (butane)

    # Add bonds between the selected carbons from mol1 and butane's terminal carbons
    edit_combined.AddBond(index1, index3, order=BondType.SINGLE)
    edit_combined.AddBond(index2, index4, order=BondType.SINGLE)

    # Get the final connected molecule and sanitize it
    connected_mol = edit_combined.GetMol()
    Chem.SanitizeMol(connected_mol)

    # Return the SMILES string of the connected molecule
    return Chem.MolToSmiles(connected_mol)

def repeat_connect_rings_C4(smiles: str, num_repeats: int) -> str:
    """
    Iteratively connects a butane molecule to the given molecule a specified number of times.

    This function uses the `connect_rings_C4` function to iteratively connect a butane molecule 
    to the provided molecule. The connection is repeated for the given number of times, or until 
    no suitable carbon atoms for connection are found.

    Parameters
    ----------
    smiles : str
        A SMILES string representing the initial molecule to be modified. The function will attempt 
        to connect butane molecules to it in the specified number of iterations.

    num_repeats : int
        The number of times the butane molecule should be connected to the initial molecule. 
        Each iteration will attempt to connect another butane molecule.

    Returns
    -------
    str
        The SMILES string representing the modified molecule after all connections have been made. 
        If at any point the connection cannot be made (e.g., no suitable carbons are found), the function 
        returns the most recent successful modification or the original molecule if no modifications were made.

    Example
    -------
    >>> repeat_connect_rings_C4("CC(C)C", 3)
    'CC(C)CCCCCCCCCCCC'  # Example output after 3 iterations of connecting butane.

    Notes
    -----
    - The function makes use of `connect_rings_C4` to handle the actual connection of butane molecules.
    - If no suitable carbon atoms for bonding are found in any iteration, the function terminates early.
    - The molecule is modified by repeatedly connecting butane molecules, and the result is returned as a SMILES string.
    """
    for _ in range(num_repeats):
        new_smiles = connect_rings_C4(smiles)
        if new_smiles is None:
            break  # Stop if no suitable carbon atom found or connection failed
        smiles = new_smiles
    return smiles

def process_smiles(smiles_list: list) -> list:
    """
    Sorts a list of SMILES strings in a zigzag pattern based on the sum of atomic numbers in each molecule.

    This function sorts a list of SMILES strings by the sum of the atomic numbers of the atoms in the molecule.
    The sorted list is then arranged in a zigzag pattern: the highest sum atomic number is placed at the first 
    position, the second highest at the second position, and so on, alternating between the highest and lowest 
    sums.

    Parameters
    ----------
    smiles_list : list of str
        A list containing SMILES strings of the molecules to analyze. The molecules will be sorted 
        based on the sum of atomic numbers of their constituent atoms.

    Returns
    -------
    list of str
        A list of SMILES strings sorted in a zigzag pattern based on the sum of atomic numbers 
        in each molecule. The list is arranged such that the highest sums are placed at even indices and 
        the lowest sums at odd indices.

    Example
    -------
    >>> smiles_list = ['CCO', 'CC(C)C', 'CCCC']
    >>> process_smiles(smiles_list)
    ['CC(C)C', 'CCCC', 'CCO']  # Example output showing zigzag sorted order

    Notes
    -----
    - The sum of atomic numbers is calculated for each molecule by summing the atomic numbers of all atoms 
      in the molecule.
    - The zigzag pattern alternates between the highest and lowest sums, placing the highest sums in even positions 
      (0-based index) and the lowest sums in odd positions.
    - The sorting ensures that the list is not simply sorted by atomic number sums in ascending or descending order, 
      but rather in a unique "zigzag" pattern.
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

def find_aldehyde_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies aldehyde carbon atoms within a given molecule.

    An aldehyde group consists of a carbonyl group (C=O) where the carbon is bonded to a hydrogen atom (–CHO).
    This function checks for carbon atoms that are part of such a group by looking for a carbon 
    bonded to one oxygen atom with a double bond and one hydrogen atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object to be analyzed. This molecule will be checked for aldehyde groups.

    Returns
    -------
    aldehyde_carbons : list
        A list of indices corresponding to carbon atoms that are part of an aldehyde group within the molecule.
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
        If no aldehyde groups are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CC(C)C=O")
    >>> find_aldehyde_carbons(mol)
    [4]  # The carbon in the aldehyde group is at index 4.

    Notes
    -----
    - The function explicitly adds hydrogen atoms to the molecule for more accurate analysis of aldehyde groups.
    - The function checks for a carbon atom bonded to one oxygen atom (double bond) and one hydrogen atom, 
      indicative of an aldehyde group.
    - Only carbon atoms that meet these criteria are included in the result list.
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

def find_hydroxy_oxygen(mol: rdchem.Mol) -> list:
    """
    Identifies the indices of hydroxy oxygen and sulfur atoms in a given RDKit molecule object.

    Hydroxy oxygen atoms are defined as oxygen atoms bonded to at least one hydrogen atom (–OH), 
    and sulfur atoms are defined as sulfur atoms bonded to at least one hydrogen atom (–SH), indicating 
    the presence of alcohol or thiol groups.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object to be analyzed. This molecule will be checked for hydroxy oxygen and sulfur atoms.

    Returns
    -------
    hydroxy_atoms : list
        A list of indices for oxygen and sulfur atoms bonded to at least one hydrogen atom,
        indicating the presence of hydroxy or thiol groups respectively. 
        Atom indices are zero-based, matching the order in which atoms appear in the RDKit molecule object.
        If no hydroxy or thiol atoms are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CC(C)O")
    >>> find_hydroxy_oxygen(mol)
    [4]  # The oxygen atom in the hydroxy group is at index 4.

    Notes
    -----
    - The function explicitly adds hydrogen atoms to the molecule for more accurate analysis of hydroxy and thiol groups.
    - Only oxygen and sulfur atoms that are bonded to at least one hydrogen atom are included in the result list.
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

def find_ketone_alpha_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies alpha carbon atoms adjacent to carbonyl carbons in ketones within a given molecule.

    Alpha carbon atoms are defined as those that are adjacent to the carbonyl group (C=O) in ketones.
    These are typically the carbons bonded to the carbonyl carbon but not directly bonded to the oxygen atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object representing the molecule to be analyzed.

    Returns
    -------
    ketone_alpha_carbons : list
        A list of indices for alpha carbon atoms that are adjacent to carbonyl carbons in ketones.
        The atom indices are zero-based, corresponding to the order in which atoms appear in the RDKit molecule object.
        If no alpha carbons are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CC(=O)C")
    >>> find_ketone_alpha_carbons(mol)
    [1]  # The alpha carbon adjacent to the carbonyl group at index 1.

    Notes
    -----
    - This function identifies ketones by finding oxygen atoms bonded to a carbonyl carbon.
    - It assumes that the carbonyl group has a double bond between the oxygen and carbon.
    - The alpha carbon is defined as a carbon atom bonded to the carbonyl carbon but not to the oxygen atom.

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

def find_alpha_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies indices of non-aromatic alpha carbon atoms that are directly bonded to aromatic carbon atoms.

    Alpha carbon atoms are defined as carbon atoms that are directly bonded to an aromatic carbon atom, 
    but are themselves not aromatic. These are typically carbons attached to aromatic rings in compounds 
    like aryl groups attached to alkyl chains.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object representing the molecule to be analyzed.

    Returns
    -------
    list
        A list of unique indices for non-aromatic alpha carbon atoms that are directly bonded to aromatic carbon atoms.
        Atom indices are zero-based, corresponding to the order in which atoms appear in the RDKit molecule object.
        If no such alpha carbons are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CC1=CC=CC=C1")
    >>> find_alpha_carbons(mol)
    [0, 3]  # The alpha carbons bonded to the aromatic ring at indices 0 and 3.

    Notes
    -----
    - This function identifies non-aromatic carbons bonded to aromatic carbons in molecules with aromatic rings.
    - The search excludes aromatic carbons, focusing only on non-aromatic carbons.
    - The set data structure is used to avoid duplicate indices for alpha carbons.

    """
    alpha_carbons = set()  # Use a set to avoid duplicate indices

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetIsAromatic():
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C' and not neighbor.GetIsAromatic():
                    alpha_carbons.add(neighbor.GetIdx())  # Add unique alpha carbon indices

    return list(alpha_carbons)  # Convert back to list for consistency with expected return type

def find_aliphatic_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies indices of aliphatic (non-aromatic) carbon atoms within a given molecule.

    Aliphatic carbon atoms are defined as carbon atoms that are not part of an aromatic ring. These include 
    carbon atoms in chains, branches, or rings that are not aromatic.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object representing the molecule to be analyzed.

    Returns
    -------
    list
        A list of indices corresponding to aliphatic carbon atoms in the molecule.
        Atom indices are zero-based, corresponding to the order in which atoms appear in the RDKit molecule object.
        If no aliphatic carbon atoms are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CC1=CC=CC=C1")
    >>> find_aliphatic_carbons(mol)
    [0, 1, 2, 3]  # The aliphatic carbons are the ones in the alkyl chain attached to the aromatic ring.

    Notes
    -----
    - This function identifies carbon atoms that are not part of any aromatic ring in the molecule.
    - Aliphatic carbons can be in straight chains, branched chains, or non-aromatic rings.
    - Aromatic carbons, which are part of rings with alternating single and double bonds (like benzene), are excluded.

    """
    aliphatic_carbons = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
            aliphatic_carbons.append(atom.GetIdx())  # Add the index of each aliphatic carbon atom

    return aliphatic_carbons

def find_benzene_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies indices of carbon atoms that are part of benzene rings in a given molecule.

    Benzene rings are identified based on the presence of six carbon atoms that are part of an aromatic system,
    where the carbon atoms are involved in alternating single and double bonds.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object representing the molecule to be analyzed.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that are part of benzene rings in the molecule.
        The indices are zero-based, corresponding to the order in which atoms appear in the RDKit molecule object.
        If no benzene rings are found, the list will be empty.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    >>> find_benzene_carbons(mol)
    [0, 1, 2, 3, 4, 5]  # The carbon atoms in the benzene ring.

    Notes
    -----
    - Benzene rings are detected by finding rings with exactly six carbon atoms that are aromatic.
    - The function returns a list of indices for the carbon atoms in the benzene ring, excluding other atoms or rings.
    - The output list may contain duplicates if the molecule contains multiple benzene rings, which are then removed by converting the list to a set.

    """
    benzene_carbons = []
    ri = mol.GetRingInfo()
    aromatic_rings = [ring for ring in ri.AtomRings() if len(ring) == 6 and all(mol.GetAtomWithIdx(idx).GetSymbol() == 'C' for idx in ring)]

    for ring in aromatic_rings:
        for idx in ring:
            if mol.GetAtomWithIdx(idx).GetIsAromatic():
                benzene_carbons.append(idx)

    return list(set(benzene_carbons))  # Remove duplicates and return the list

def find_ban_carbons(mol: rdchem.Mol) -> list:
    """
    Identifies carbon atoms bonded to exactly two hydrogen atoms and not connected to any other carbons.

    This function is designed to find carbon atoms in a molecule that are bonded to exactly two hydrogen atoms 
    and are not bonded to other carbon atoms (i.e., the carbon is isolated or terminal, not part of a longer chain).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        An RDKit molecule object representing the molecule to be analyzed, before adding explicit hydrogens.

    Returns
    -------
    list
        A list of indices corresponding to carbon atoms that meet the criteria:
        - Bonded to exactly two hydrogen atoms.
        - Not connected to any other carbon atoms.

    Example
    -------
    >>> mol = Chem.MolFromSmiles("CH3CH2OH")
    >>> find_ban_carbons(mol)
    [0, 3]  # For a molecule like ethanol, the carbon atoms bonded to two hydrogens are indexed.

    Notes
    -----
    - This function first adds explicit hydrogens to the molecule to ensure the number of hydrogen atoms can be counted accurately.
    - It considers only carbon atoms that are bonded to exactly two hydrogens and are not connected to other carbons, excluding carbon-carbon bonds.

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

    # Find carbon atoms only connected to hydrogen atoms (excluding those connected to other carbons)
    single_linked_carbons = [
        idx for idx, neighbors in neighbor_atoms.items()
        if not any(neighbor in carbon_atoms_with_two_hydrogen for neighbor in neighbors)
    ]
    
    return single_linked_carbons

def connect_molecules(smiles1: str, smiles2: str) -> str:
    """
    Connects two molecules at suitable bonding positions, based on various functional groups.

    Parameters
    ----------
    smiles1 : str
        The SMILES string representing the first molecule.
    smiles2 : str
        The SMILES string representing the second molecule to be connected.

    Returns
    -------
    str
        The SMILES string representing the combined molecule if a valid connection is made,
        or None if no valid connection can be found.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # Define lists of functions to identify suitable connection points on each molecule
    connection_functions1: list = [
        find_aldehyde_carbons,
        find_hydroxy_oxygen,
        find_ketone_alpha_carbons,
        find_alpha_carbons, 
        find_aliphatic_carbons,
        find_benzene_carbons
    ]
    
    connection_functions2: list = [
        find_aldehyde_carbons,
        find_hydroxy_oxygen,
        find_ketone_alpha_carbons,
        find_alpha_carbons,
        find_aliphatic_carbons,
        find_benzene_carbons
    ]
    
    # Outer loop to iterate through possible connection points on mol1
    for find_atoms1 in connection_functions1:
        atoms1 = find_atoms1(mol1)
        atoms1 = [idx for idx in atoms1 if len([neighbor for neighbor in mol1.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetSymbol() != 'H']) < 3]
        atoms1 = [idx for idx in atoms1 if idx not in find_ban_carbons(mol1)]
        
        if atoms1:
            # Inner loop to iterate through possible connection points on mol2
            for find_atoms2 in connection_functions2:
                atoms2 = find_atoms2(mol2)
                atoms2 = [idx for idx in atoms2 if len([neighbor for neighbor in mol2.GetAtomWithIdx(idx).GetNeighbors() if neighbor.GetSymbol() != 'H']) < 3]
                atoms2 = [idx for idx in atoms2 if idx not in find_ban_carbons(mol2)]
                
                # Ensure no connection between two hydroxy groups
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

def show_ring_carbon_numbers(smiles: str) -> list:
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

def select_carbons_from_different_rings(current_molecule: str, ring_carbons: list, n: int) -> list:
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

def count_atoms(smiles_input: str or list) -> dict:
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

def balance_c_and_n_atoms(smiles: str, target_element_counts: dict) -> str:
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

def find_atom_indices(smiles: str, atom_symbol: str) -> list:
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

def balance_o_and_s_atoms(smiles: str, target_element_counts: dict) -> str:
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

def calculate_unsaturated_carbon_ratio(smiles: str) -> float:
    """
    Calculates the ratio of unsaturated carbon atoms (participating in double or triple bonds) 
    to the total number of carbon atoms in a molecule represented by a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.

    Returns
    -------
    float
        The ratio of unsaturated carbon atoms to total carbon atoms in the molecule.
        Returns 0 if there are no carbon atoms in the molecule or if the input SMILES is invalid.
    """
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
    Counts specified elements and their types (e.g., unsaturated or aliphatic carbons) in one or more SMILES strings.
    
    Specifically counts:
    - 'C_N_ar': Unsaturated carbon or nitrogen atoms (those participating in double or triple bonds).
    - 'C_al': Aliphatic (single-bonded) carbon atoms.
    - 'O_S': Oxygen and sulfur atoms.
    - 'H': Hydrogen atoms.

    Parameters
    ----------
    smiles_input : str or list
        A single SMILES string or a list of SMILES strings representing one or more molecules.

    Returns
    -------
    dict
        A dictionary with the counts of each property: {'C_N_ar', 'C_al', 'O_S', 'H'}.
        The counts reflect the number of atoms of each type found in the provided molecule(s).
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
                # Check if the atom is part of an unsaturated structure (double or triple bond)
                if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()):
                    property_counts['C_N_ar'] += 1  # Count as unsaturated C (or N)
                else:
                    property_counts['C_al'] += 1  # Count as aliphatic carbon
            elif atom.GetSymbol() in ['O', 'S']:
                property_counts['O_S'] += 1  # Count oxygen and sulfur atoms
            elif atom.GetSymbol() == 'H':
                property_counts['H'] += 1  # Count hydrogen atoms

    return property_counts

def drawMolecule(smiles: str) -> None:
    """
    Draws a molecule from a given SMILES string and displays the image.
    
    This function uses RDKit to create a molecule object from the provided SMILES string
    and then generates an image of the molecule, which is displayed using the default image viewer.

    Parameters
    ----------
    smiles : str
        The SMILES string representing the molecule to be drawn.

    Returns
    -------
    None
        This function does not return a value, it directly displays the molecule's image.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Create the molecule without sanitization
    img = Draw.MolToImage(mol, size=(600, 600))  # Generate the image
    img.show()  # Display the image

def drawMolecules(smiles_list: list, molsPerRow: int, maxMols: int = 100) -> list:
    """
    Draws a grid image of molecules from a list of SMILES strings.

    This function takes a list of SMILES strings, converts them into RDKit molecule objects, and generates a grid image.
    It also provides a summary of the molecular properties (C, N, O, H counts) for each molecule in the grid.

    Parameters
    ----------
    smiles_list : list of str
        A list of SMILES strings representing the molecules to be drawn.
        
    molsPerRow : int
        The number of molecules to display per row in the grid image.
        
    maxMols : int, optional, default=100
        The maximum number of molecules to display in the grid. If more molecules are provided, they will be truncated.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing molecular property counts (C, N, O, H) for each molecule.
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

def calculate_mass_percentages(atom_counts: dict) -> dict:
    """
    Calculates the mass percentages of elements based on their counts in a molecule.

    This function uses the atomic masses of carbon, hydrogen, oxygen, nitrogen, and sulfur to compute
    the mass percentages of each element in the molecule, given the counts of each element.

    Parameters
    ----------
    atom_counts : dict of {str: int}
        A dictionary with element symbols as keys (e.g., 'C', 'H', 'O') and their counts in the molecule as values.

    Returns
    -------
    dict of {str: float}
        A dictionary with element symbols as keys and their mass percentages as values.
    """
    # Atomic masses in atomic mass units (amu)
    atomic_masses = {
        'C': 12.01,
        'H': 1.008,
        'O': 16.00,
        'N': 14.01,
        'S': 32.06
    }
    
    # Calculate the total mass for each element
    masses = {element: count * atomic_masses[element] for element, count in atom_counts.items()}
    
    # Calculate the total mass of the molecule
    total_mass = sum(masses.values())
    
    # Calculate the mass percentages of each element
    mass_percentages = {element: (mass / total_mass) * 100 for element, mass in masses.items()}
    
    return mass_percentages

def ad2daf(C: float, N: float, H: float, S: float, M: float, A: float) -> tuple:
    """
    Converts the element percentages from the AD (as-determined) basis to the DAF (dry, ash-free) basis.

    This function calculates the percentage of each element (C, N, H, S, and O) on the DAF basis,
    given their values on the AD basis.

    Parameters
    ----------
    C : float
        Carbon percentage in the AD basis.
    N : float
        Nitrogen percentage in the AD basis.
    H : float
        Hydrogen percentage in the AD basis.
    S : float
        Sulfur percentage in the AD basis.
    M : float
        Moisture percentage (unused in calculations but might be part of input data).
    A : float
        Ash percentage (unused in calculations but might be part of input data).

    Returns
    -------
    tuple of float
        A tuple containing the percentages of Carbon (C), Nitrogen (N), Hydrogen (H), Sulfur (S),
        and Oxygen (O) on the DAF basis.
    """
    # Calculate the oxygen percentage in AD basis
    O = 100 - (C + N + H + S + M + A)
    
    # Calculate the total sum of C, N, H, S, and O on AD basis
    total_CNHSO_ad = C + N + H + S + O
    print('O_ad=', O)
    
    # Calculate the percentage of each element on DAF basis
    C_daf = C / total_CNHSO_ad * 100
    N_daf = N / total_CNHSO_ad * 100
    H_daf = H / total_CNHSO_ad * 100
    S_daf = S / total_CNHSO_ad * 100
    O_daf = O / total_CNHSO_ad * 100
    
    return C_daf, N_daf, H_daf, S_daf, O_daf

def daf2ad(C_daf: float, N_daf: float, H_daf: float, S_daf: float, O_daf: float, M: float, A: float) -> tuple:
    """
    Converts the element percentages from the DAF (dry, ash-free) basis to the AD (as-determined) basis.

    This function calculates the percentage of each element (C, N, H, S, and O) on the AD basis,
    given their values on the DAF basis, and considering moisture and ash content.

    Parameters
    ----------
    C_daf : float
        Carbon percentage in the DAF basis.
    N_daf : float
        Nitrogen percentage in the DAF basis.
    H_daf : float
        Hydrogen percentage in the DAF basis.
    S_daf : float
        Sulfur percentage in the DAF basis.
    O_daf : float
        Oxygen percentage in the DAF basis.
    M : float
        Moisture percentage.
    A : float
        Ash percentage.

    Returns
    -------
    tuple of float
        A tuple containing the percentages of Carbon (C), Nitrogen (N), Hydrogen (H), Sulfur (S),
        and Oxygen (O) on the AD basis.

    Example
    -------
    Given the following input data:
    
    C_daf = 83.96
    N_daf = 3.81
    H_daf = 3.31
    S_daf = 0.39
    O_daf = 8.99
    M = 2.51  # Moisture
    A = 1.66  # Ash

    The function is called as:

    C_ad, N_ad, H_ad, S_ad, O_ad = daf2ad(C_daf, N_daf, H_daf, S_daf, O_daf, M, A)

    The output will be:

    ad基准下百分比: C_ad, N_ad, H_ad, S_ad, O_ad
    CNHSOMA和: C_ad + N_ad + H_ad + S_ad + O_ad + M + A
    """
    # Conversion factor for converting from DAF to AD
    conversion_factor = (100 - (M + A)) / 100
    
    # Calculate the element percentages on AD basis
    C_ad = C_daf * conversion_factor
    N_ad = N_daf * conversion_factor
    H_ad = H_daf * conversion_factor
    S_ad = S_daf * conversion_factor
    
    # Calculate the total sum of C, N, H, S on AD basis
    total_CNHS_ad = C_ad + N_ad + H_ad + S_ad
    
    # Calculate oxygen percentage on AD basis
    O_ad = 100 - (total_CNHS_ad + M + A)
    
    return C_ad, N_ad, H_ad, S_ad, O_ad


def calculate_MA(C_ad: float, N_ad: float, H_ad: float, S_ad: float, C_daf: float, O_daf: float) -> float:
    """
    Calculates the sum of moisture (M) and ash (A) content based on the element percentages 
    in the AD (as-determined) basis and DAF (dry, ash-free) basis for a given material.

    This function computes the total moisture and ash content (M + A) based on the known 
    values of carbon, nitrogen, hydrogen, sulfur, and oxygen in both AD and DAF forms.

    Parameters
    ----------
    C_ad : float
        Carbon percentage in the AD basis.
    N_ad : float
        Nitrogen percentage in the AD basis.
    H_ad : float
        Hydrogen percentage in the AD basis.
    S_ad : float
        Sulfur percentage in the AD basis.
    C_daf : float
        Carbon percentage in the DAF basis.
    O_daf : float
        Oxygen percentage in the DAF basis.

    Returns
    -------
    float
        The sum of moisture and ash content (M + A) on the AD basis.

    Example
    -------
    Given the following input data:

    C_ad = 80.92
    N_ad = 3.49   # Nitrogen
    H_ad = 3.03   # Hydrogen
    S_ad = 0.38   # Sulfur
    C_daf = 83.86
    O_daf = 8.99

    The function is called as:

    MA_sum = calculate_MA(C_ad, N_ad, H_ad, S_ad, C_daf, O_daf)

    The output will be:

    MA_sum: 8.29
    """
    # Calculate the total CNHS percentage on the AD basis
    total_CNHS_ad = C_ad + N_ad + H_ad + S_ad
    
    # Calculate O_ad based on the ratio between C_ad and C_daf
    O_ad = (C_ad / C_daf) * O_daf
    
    # Calculate the total sum of moisture (M) and ash (A)
    MA_sum = 100 - (total_CNHS_ad + O_ad)
    
    return MA_sum
