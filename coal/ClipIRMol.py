import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.Draw import MolsToGridImage
from scipy.interpolate import CubicSpline
import torch
import torch.nn.functional as F
import pickle
import coal.main as main
import random
from IPython.display import display
from rdkit.Chem.rdchem import BondType

# Globel variables
MODEL_FILENAME = 'clip_model_60000'
MODEL_PARAMS_FILENAME = MODEL_FILENAME + '_parameters.pth'
TEXT_ENCODER_PARAMS_PATH = 'model_2023-04-14_TextEncoder_parameters.pkl'
TEXT_FEATURES_PATH = "text_features.npy"


def check_isotope(molecule):
    for atom in molecule.GetAtoms():
        if atom.GetIsotope() != 0:
            return False
    return True

def check_c_neigh_o(molecule):
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'C':
            oxygen_neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == 'O']
            if len(oxygen_neighbors) > 1:
                return False
    return True

def check_rings(molecule):
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

def check_six_membered_ring(molecule, ring):
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    double_bonds = sum(1 for atom in ring_atoms if atom.GetHybridization() == rdchem.HybridizationType.SP2)
    n_count = sum(1 for atom in ring_atoms if atom.GetSymbol() == 'N')
    if not molecule.GetRingInfo().IsAtomInRingOfSize(ring_atoms[0].GetIdx(), 6) or double_bonds != 6 or n_count > 1:
        return False

    for atom in ring_atoms:
        if atom.GetSymbol() == 'N' and atom.GetTotalDegree() == 3:
            return False

    return True

def check_five_membered_ring(molecule, ring):
    ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
    double_bonds = sum(1 for atom in ring_atoms if atom.GetHybridization() == rdchem.HybridizationType.SP2)
    ns_count = sum(1 for atom in ring_atoms if atom.GetSymbol() in ['N', 'S'])
    return double_bonds == 4 and ns_count <= 1

def check_ring_no_oxy_sulph(molecule):
    rings = molecule.GetRingInfo().AtomRings()  
    for ring in rings:
        ring_atoms = [molecule.GetAtomWithIdx(idx) for idx in ring]
        oxy_sulph_count = sum(atom.GetSymbol() in ['N', 'S'] for atom in ring_atoms)
        
        # Check for oxygen atoms in the ring
        for idx in ring:
            atom = molecule.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'O':
                return False
        
        if oxy_sulph_count > 1:
            return False
    return True

def check_invalid_bonds(molecule):
    for bond in molecule.GetBonds():
        bond_atoms = set([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()])
        invalid_combinations = [{'N', 'O'}, {'O', 'O'}, {'S', 'O'}, {'S', 'S'}]
        if bond_atoms in invalid_combinations:
            return False
    return True

def check_chain_double_bonds(molecule):
    for bond in molecule.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
            atom1_in_ring = any(molecule.GetRingInfo().IsAtomInRingOfSize(atom1.GetIdx(), ring_size) for ring_size in range(3, molecule.GetNumAtoms() + 1))
            atom2_in_ring = any(molecule.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), ring_size) for ring_size in range(3, molecule.GetNumAtoms() + 1))
            if not (atom1_in_ring and atom2_in_ring) and bond.GetBondType() != rdchem.BondType.SINGLE:
                return False
    return True

def check_chain_n_c(molecule):
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'N':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    if not (molecule.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), 6) and molecule.GetRingInfo().IsAtomInRingOfSize(neighbor.GetIdx(), 6)):
                        return False
    return True

def satisfy_conditions(molecule):
    return (
        check_isotope(molecule) and
        check_c_neigh_o(molecule) and
        check_rings(molecule) and
        check_ring_no_oxy_sulph(molecule) and
        check_invalid_bonds(molecule) and
        check_chain_double_bonds(molecule) and
        check_chain_n_c(molecule)
    )


def process_image(image_path):
    df = pd.read_csv(image_path)
    if not all(df['X'].diff()[1:] > 0):
        df = df.sort_values(by='X', ascending=True)

    x_data = df['X'].values
    y_data = df['Y'].values

    # cubic spline interpolation
    cs = CubicSpline(x_data, y_data)
    x = np.arange(400, 4000, 2)
    y = cs(x)

    data = {'X': x, 'Y': y}
    selected_df = pd.DataFrame(data)

    selected_data = selected_df.astype('float32').to_numpy()
    return selected_data.reshape(2, 1, 36, 50)

def load_model_and_features():
    with open(TEXT_ENCODER_PARAMS_PATH, 'rb') as f:
        text_encoder_params = pickle.load(f)

    text_encoder = main.TextEncoder(text_encoder_params)
    loaded_model_params = torch.load(MODEL_PARAMS_FILENAME)
    loaded_clip_model = main.CLIP(text_encoder)
    loaded_clip_model.load_state_dict(loaded_model_params)

    loaded_text_features = np.load(TEXT_FEATURES_PATH)

    return loaded_clip_model, loaded_text_features

def compute_similarity(image_input, clip_model, text_features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device).eval()

    image_input = torch.from_numpy(image_input).float().to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()

    image_features = image_features[0].unsqueeze(0).cpu().numpy()

    normalized_image_features = F.normalize(torch.from_numpy(image_features), dim=1).numpy()
    normalized_text_features = F.normalize(torch.from_numpy(text_features), dim=1).numpy()

    similarity_scores = 100.0 * np.dot(normalized_image_features, normalized_text_features.T)
    return similarity_scores

# 根据相似度得分对一系列分子进行排序，然后选择满足特定条件的前n个分子
def skip_and_update_top_n_ranking(similarity_scores, top_n, df):
    # 空列表存储分子和对应的相似度
    updated_predicted_smiles_and_scores = []
    
    for i in range(similarity_scores.shape[0]):
        valid_smiles_list = []
        # 获取按相似度排序的分子索引
        sorted_indices = np.argsort(-similarity_scores[i])

        # 遍历这些排序索引
        for index in sorted_indices:
            # 获取对应索引的分子的SMILES表示
            smiles = list(df['smiles'])[index]
            # 将SMILES表示转化为分子对象
            molecule = Chem.MolFromSmiles(smiles)
            
            all_rings_satisfy_conditions = True
            all_rings_satisfy_conditions = satisfy_conditions(molecule)
            if all_rings_satisfy_conditions:
                score = similarity_scores[i, index]
                valid_smiles_list.append((smiles, score))
                
                # 如果满足条件的分子数量达到了我们希望的数量（前n个）
                if len(valid_smiles_list) == top_n:
                    break
        # 将满足条件的分子列表添加到最终的列表中
        updated_predicted_smiles_and_scores.append(valid_smiles_list)
    
    # 包含了每个分子的前n个满足条件的分子以及它们的相似性分数
    return updated_predicted_smiles_and_scores

def retrieve_small_molecules(image_path):
    df = pd.read_csv('smile_ir.csv')
    image_input = process_image(image_path)
    loaded_clip_model, loaded_text_features = load_model_and_features()

    similarity_scores = compute_similarity(image_input, loaded_clip_model, loaded_text_features)

    top_n = 40
    updated_predicted_smiles_and_scores = skip_and_update_top_n_ranking(similarity_scores, top_n, df)

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

def convert_data_to_smiles_scores(retrieved_molecules):
    """Convert the retrieved molecules data into a list of (SMILES, score) tuples."""
    return [(entry['smiles'], entry['score']) for entry in retrieved_molecules]

def display_molecules(smiles_and_scores, mols_per_row=5, img_size=(300, 300)):
    """Create and display a grid image of molecules with their associated scores."""
    
    # Convert SMILES strings to RDKit molecule objects
    predicted_molecules = [Chem.MolFromSmiles(smiles) for smiles, _ in smiles_and_scores]
    
    # Create legends for each molecule
    legends = [f"{i+1}. {smiles} (score: {score:.4f})" for i, (smiles, score) in enumerate(smiles_and_scores)]
    
    # Create the grid image
    img = MolsToGridImage(predicted_molecules, molsPerRow=mols_per_row, subImgSize=img_size, legends=legends)
    
    display(img)
    return 

def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

# 环状结构都有哪些地方能连
def find_required_carbons(smiles):
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

def connect_rings_C4(smiles1):
    smiles2 = 'CCCC'
    carbons2 = [0, 3]

    carbons1 = find_required_carbons(smiles1)

    # Check if no suitable carbon atom was found in the molecule
    if not carbons1:
        print("No suitable carbon atom found in the molecule")
        return None

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # 寻找smiles1中所有相邻的满足条件的碳原子对。
    index_pairs_carbons1 = []
    
    for i in range(len(carbons1)):
        atom1 = mol1.GetAtomWithIdx(carbons1[i])
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIdx() in carbons1:
                if carbons1[i] < neighbor.GetIdx():
                    index_pairs_carbons1.append((carbons1[i], neighbor.GetIdx()))

    # 随机选择一对相邻的满足条件的碳原子。
    chosen_pair1 = random.choice(index_pairs_carbons1)
#     print('chosen_pair1', chosen_pair1)

    # Combine molecules
    combined = Chem.CombineMols(mol1, mol2)
    
    # Create an editable molecule
    edit_combined = Chem.EditableMol(combined)

    # The indices will change after combination, recalculate them
    index1 = chosen_pair1[0]  # this is for the mol1
    index2 = chosen_pair1[1]  # this is for the mol1
    index3 = len(mol1.GetAtoms())  # this is for the first atom of mol2
    index4 = len(mol1.GetAtoms()) + 3  # this is for the last atom of mol2

    # Define a set to store the indices of all atoms in mol2 (butane) in the merged molecule
    butane_indices = set(range(len(mol1.GetAtoms()), len(mol1.GetAtoms()) + 4))
    butane_indices.add(index1)
    butane_indices.add(index2)
#     print(butane_indices)
    
    # Add the bonds
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

    
    # Clean up the molecule
    Chem.SanitizeMol(connected_mol)

    return Chem.MolToSmiles(connected_mol)

def find_required_aldehyde_carbons(smiles):
    # 从SMILES字符串创建分子对象并添加氢
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # 存储满足条件的碳原子的索引
    carbons = []

    # 遍历分子中的每个原子
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            neighbors = atom.GetNeighbors()
            
            # 查找有一个C、一个H和一个O的邻居的原子
            if len(neighbors) == 3:
                symbols = [neighbor.GetSymbol() for neighbor in neighbors]
                if symbols.count('C') == 1 and symbols.count('H') == 1 and symbols.count('O') == 1:
                    carbons.append(atom.GetIdx())

    return carbons

# 羰基连接的贝塔C的邻居至少要有一个的邻居数量为3以下，那么返回False
def satisfy_beta_carbons_conditions(smiles):
    # 从SMILES字符串创建分子对象并添加氢
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

    # 获取羰基碳的索引
    aldehyde_carbons = find_required_aldehyde_carbons(smiles)

    # 遍历每一个羰基碳
    for aldehyde_carbon in aldehyde_carbons:
        # 获取这个羰基碳的阿尔法碳
        alpha_carbons = [neighbor for neighbor in mol.GetAtomWithIdx(aldehyde_carbon).GetNeighbors() if neighbor.GetSymbol() == 'C']

        # 如果没有阿尔法碳，返回 False
        if not alpha_carbons:
            return False

        # 遍历每一个阿尔法碳
        for alpha_carbon in alpha_carbons:
            # 获取这个阿尔法碳的所有贝塔碳，并检查是否有氢连接
            beta_carbons = [neighbor for neighbor in alpha_carbon.GetNeighbors() if neighbor.GetSymbol() == 'C']
            beta_carbons = [beta for beta in beta_carbons if beta.GetIdx() != aldehyde_carbon]  # 去除阿尔法碳索引
            if not any(['H' in [n.GetSymbol() for n in beta.GetNeighbors()] for beta in beta_carbons]):
                return False
    return True

def filter_smiles_and_scores(updated_predicted_smiles_and_scores):
    # 存储满足条件的元组
    filtered_smiles_and_scores = []
    
    for smiles_and_score in updated_predicted_smiles_and_scores[0]:
        # 获取SMILES字符串和分数
        smiles, score = smiles_and_score
        
        # 判断是否满足贝塔碳的条件
        if satisfy_beta_carbons_conditions(smiles):
            filtered_smiles_and_scores.append((smiles, score))
    
    return filtered_smiles_and_scores