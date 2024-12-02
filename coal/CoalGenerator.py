import random
from rdkit import Chem
import coal.utils as ut

class CoalGenerator:
    def __init__(self, data):
        # Initialize instance variables including the lists of compounds and chemical properties
        self.type = data['type']
        self.coal_smiles_list = data['coal_smiles_list']
        self.extra_smiles_list = data['extra_smiles_list']
        self.ele_ratio = data['ele_ratio']
        self.C90 = data['C90']
        self.C180 = data['C180']
        self.C_moles = data['C_atom']
        self.total_smiles_list = self.coal_smiles_list + self.extra_smiles_list
        self.target_atomNum = {}
        self.carbonyl = 0
        self.hydroxyl = 0
        # predict
        self.predicted_atomNum = {}
        self.candidate_smiles_list = []
        self.topN_smiles_list = {}


    def run(self):
        # Stage 1: generate candidate SMILES (Substructures)
        # Step 1: Generate Top N candidate molecules and their target attributes
        topN_smiles_list = ut.getPackage(self.total_smiles_list)
        topN_property_list = ut.drawMolecules(topN_smiles_list, molsPerRow=4)
        # Step 2: Calculate the target molecule's chemical formula and target attributes
        # Calculate carbonyl and hydroxyl
        self.target_atomNum, adjusted_target_atomNum = ut.getTarget(self.C90, self.ele_ratio, self.C_moles, self.carbonyl, self.hydroxyl)
        
        print(f"Target {self.type} coal model with molecular formula C{self.target_atomNum['C']}H{self.target_atomNum['H']}O{self.target_atomNum['O']}N{self.target_atomNum['N']}S{self.target_atomNum['S']}")
        print(f"Target attribute: {adjusted_target_atomNum}")

        # Step 3: Calculate all combinations that meet the target C_N_ar
        nested_dict_H = ut.build_H_nested_dict(topN_property_list)
        cn_list = ut.find_combinations(topN_property_list, adjusted_target_atomNum['C_N_ar'])
        print(f"All combinations that meet the target C_N_ar: {cn_list}")
        
        # Step 4: Calculate the combinations that meet not only C_N_ar but also target O_S
        all_final_selections = []
        count = 1

        for selection_dic in cn_list:
            # print("Screening", selection_dic, adjusted_target_atomNum['O_S'])
            print(f"Evaluating combination {count}/{len(cn_list)}")
            # 单核默认版本
            # final_nested_dict_count_list = ut.backtrack_combinations(nested_dict_H, selection_dic, adjusted_target_atomNum['O_S'], max_depth=50)
            # 并行版本
            final_nested_dict_count_list = ut.parallel_backtrack_combinations(nested_dict_H, selection_dic, adjusted_target_atomNum['O_S'], max_depth=50, n_jobs=40)


            all_final_selections.extend(final_nested_dict_count_list)
            count += 1

        # test
        # selection_dic = {0: 0, 14: 37, 15: 0, 16: 0, 18: 3}
        # final_nested_dict_count_list = ut.parallel_backtrack_combinations(nested_dict_H, selection_dic, adjusted_target_atomNum['O_S'], max_depth=50, n_jobs=40)
        # all_final_selections.extend(final_nested_dict_count_list)

        # Step 5: Find the combination with the minimum total hydrogen
        total_H_list = []
        for selection in all_final_selections:
            total_H = 0
            for H_key, inner_dict in selection.items():
                for O_S_key, count in inner_dict.items():
                    # Assuming nested_dict_H is structured similarly and contains relevant multipliers or counts
                    if O_S_key in nested_dict_H[H_key]:
                        total_H += count * nested_dict_H[H_key][O_S_key]
            total_H_list.append(total_H)

        if total_H_list:
            min_H_index = total_H_list.index(min(total_H_list))
            min_H_selection = all_final_selections[min_H_index]
            print("The selection with the minimum total H is:", min_H_selection)
        else:
            min_H_selection = {}
            print("No valid selections were found.")

        # Step 6: Generate candidate SMILES list and predict each atom counts with minimum total H counts
        self.candidate_smiles_list, self.predicted_atomNum = ut.generate_candidate_smiles(min_H_selection, topN_property_list, topN_smiles_list, adjusted_target_atomNum['C_al'])
        print("After DFS get a Combination: ")
        print('Candidate molecules respresenting in SMILES:', self.candidate_smiles_list)
        print('Total atom counts:', self.predicted_atomNum)
        ut.drawMolecules(self.candidate_smiles_list, molsPerRow=4)
        
        # Stage 2: generate aromatic nucleus SMILES
        # Muti-ring molecules (>=1)
        smiles_component_list1 = []
        # -CH3, C=O, -OH
        smiles_component_list2 = []

        for smiles in self.candidate_smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            num_rings = len(Chem.GetSymmSSSR(mol))
            if num_rings >= 1:
                smiles_component_list1.append(smiles)
            else:
                smiles_component_list2.append(smiles)

        # step 1: smiles_component_list1 + C3 
        print('加丙烷前后')
        print(smiles_component_list1, smiles_component_list2)
        smiles_component_list1, smiles_component_list2 = ut.update_smiles_lists(smiles_component_list1, smiles_component_list2)
        print(smiles_component_list1, smiles_component_list2)

        ut.drawMolecules(smiles_component_list1, molsPerRow=4)

        # step 2: smiles_component_list1 + C3 + ring-ring-2C
        result_list = [(smiles, ut.find_required_carbons(smiles)) for smiles in smiles_component_list1]
        sub_result_list = []

        # 从 result_list 选择配对并生成 sub_result_list
        while len(result_list) > 1:
            elem1 = random.choice(result_list)
            result_list.remove(elem1)
            elem2 = random.choice(result_list)
            result_list.remove(elem2)
            sub_result_list.append([elem1, elem2])

        if result_list:
            sub_result_list.append([result_list[0]])

        # 生成最终连接的环结构
        ring_list = []
        total_C_added = 0  # 初始化添加的'C'的总数

        for sub_list in sub_result_list:
            before_connect = ut.count_atoms([item[0] for item in sub_list])  # 计算连接前的碳原子总数
            final_connected_smiles = ut.connect_rings(sub_list)
            after_connect = ut.count_atoms([final_connected_smiles])  # 计算连接后的碳原子总数
            print(f"Before connect: {before_connect}, After connect: {after_connect}")
            
            # 计算每次连接实际增加的碳原子数
            C_added = after_connect['C'] - before_connect['C']
            total_C_added += C_added
            ring_list.append(final_connected_smiles)

        # 计算需要移除的碳原子总数
        total_C_to_remove = max(0, total_C_added)

        # 调整第二列表中的组分，确保移除相应数量的'C'
        methane_count = smiles_component_list2.count('C')
        smiles_component_list2 = [smile for smile in smiles_component_list2 if smile != 'C']
        methane_count_to_keep = max(0, methane_count - total_C_to_remove)  # 确保不会有负数出现
        smiles_component_list2.extend(['C'] * methane_count_to_keep)

        print(f"Remaining 'C' in list2: {smiles_component_list2.count('C')}")
        print(f'methane_count:{methane_count}')

        # 最终组合
        smiles_component_list = ring_list + smiles_component_list2

        # 输出最终的原子总数
        print(f'SMILES component list: {ut.count_atoms(smiles_component_list)}')

        # step 3: combine smiles_component_list2 to smiles_component, and adjust the combined smiles_component_list
        # Compute how many 'C' can be replaced by 'CCCC'
        count_replaced = smiles_component_list.count('C') // 4

        # Replace 'C' by 'CCCC' for count_replaced times
        for _ in range(count_replaced):
            idx = smiles_component_list.index('C')  # Find the first 'C'
            smiles_component_list[idx] = 'CCCC'  # Replace it

            # Remove three additional 'C' to keep the same number of atoms
            for _ in range(3):
                smiles_component_list.remove('C')

        # 使用此函数计算smiles_component_list中的羟基氧的总数
        total_hydroxy_oxygen = ut.count_hydroxy_oxygen(smiles_component_list) + ut.count_ketone_carbons(smiles_component_list)
        # print("The total number of hydroxy oxygen atoms in the list is:", total_hydroxy_oxygen)

        # 计算除 'CCCC' 以外的其他分子的数量
        other_molecules_count = len(smiles_component_list) - smiles_component_list.count('CCCC')
        # print("The number of other molecules (except 'CCCC') in the list is:", other_molecules_count)

        # 检查 除了丁烷以外的分子数量 和 羟基 的差值
        difference = other_molecules_count - total_hydroxy_oxygen

        while difference < 0:
            # 如果差值小于0，则将一个 'CCCC' 替换为四个 'C'
            try:
                idx = smiles_component_list.index('CCCC')  # Find the first 'CCCC'
                smiles_component_list.pop(idx)  # Remove the 'CCCC'
                for _ in range(4):  # Add four 'C' into the list
                    smiles_component_list.append('C')
                
                # 重新计算other_molecules_count和difference
                other_molecules_count = len(smiles_component_list) - smiles_component_list.count('CCCC')
                difference = other_molecules_count - total_hydroxy_oxygen
            except ValueError:
                print("No 'CCCC' found in the list to be replaced.")
                break  # If no 'CCCC' can be found, break the loop

        # step 4: sorted smiles_component_list, namely smiles_component_list_ordered 
        # 将smiles_component_list列表分成三个部分：复杂的分子，丁烷和甲烷
        complex_molecules = [smiles for smiles in smiles_component_list if smiles not in ['CCCC', 'C']]
        butane = [smiles for smiles in smiles_component_list if smiles == 'CCCC']
        methane = [smiles for smiles in smiles_component_list if smiles == 'C']

        # 将复杂的分子各自单独作为列表
        complex_molecules_lists = [[molecule] for molecule in complex_molecules]

        # 将丁烷依次分配给这些列表
        for molecule_list in complex_molecules_lists:
            if butane:
                molecule_list.append(butane.pop())

        # 如果有多的丁烷，再依次添加到这些列表
        while butane:
            for molecule_list in complex_molecules_lists:
                if butane:
                    molecule_list.append(butane.pop())
                else:
                    break

        # 对每一个 molecule_list 进行处理
        smiles_component_list_updated = []

        for molecule_list in complex_molecules_lists:
            # 计算 molecule_list 中 'CCCC' 的数量
            num_butane = molecule_list.count('CCCC')

            # 对 molecule_list 的第一个元素（复杂分子）重复应用 `connect_rings_C4`
            updated_smiles = ut.repeat_connect_rings_C4(molecule_list[0], num_butane)

            # 添加更新后的分子到新的列表中
            smiles_component_list_updated.append(updated_smiles)

        # 将甲烷添加到新的分子列表中
        smiles_component_list_updated.extend(methane)

        # print('smiles_component_list_updated',smiles_component_list_updated)

        for smiles in smiles_component_list_updated:
            counts = {'C': 0, 'N': 0, 'H': 0, 'S': 0, 'O': 0}
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in counts:
                    counts[atom.GetSymbol()] += 1
        #     print(f"For SMILES string {smiles}, element counts are: {counts}")
            
        smiles_component_list_ordered = ut.process_smiles(smiles_component_list_updated)

        # ut.drawMolecules(smiles_component_list_ordered, molsPerRow=4)
        print('连接前的检查', ut.count_atoms(smiles_component_list_ordered))

        # 取出第一个分子
        current_molecule = smiles_component_list_ordered[0]
        # 遍历剩余分子
        for smiles in smiles_component_list_ordered[1:]:
            while True:
                connected_smiles = ut.connect_molecules(current_molecule, smiles)

                if connected_smiles is None:
                    print(f"Unable to connect {current_molecule} with {smiles}")
                    break
                else:
                    connected_mol = Chem.MolFromSmiles(connected_smiles)
                    if connected_mol is not None:
        #                 print(f"Connected {current_molecule} with {smiles}: \n{connected_smiles}")
                        # 将连接后的分子作为当前分子
                        current_molecule = connected_smiles
                        break
                    else:
                        print(f"Invalid SMILES generated: {connected_smiles}. Retrying...")

        # Initialize variables
        current_smiles = current_molecule

        # C和N的互换
        # current_smiles = None
        # max_attempts = 10
        # attempt = 0

        # # Loop until the SMILES string is successfully modified or the number of attempts exceeds the maximum
        # while current_smiles is None and attempt < max_attempts:
        #     attempt += 1
        #     try:
        #         modified_smiles = ut.balance_c_and_n_atoms(current_molecule, self.target_atomNum)
        #         if Chem.MolFromSmiles(modified_smiles) is not None:
        #             current_smiles = modified_smiles
        #         else:
        #             print(f"Invalid SMILES generated at attempt {attempt}: {modified_smiles}")
        #     except ValueError as e:
        #         if str(e) == "Sample larger than population or is negative":
        #             print("Not enough atoms to replace. Please adjust the target element counts.")
        #             break
        #         else:
        #             print(f"Exception caught at attempt {attempt}: {e}")
        #     except Exception as e:
        #         print(f"Unexpected exception at attempt {attempt}: {e}")
        
        # O和S的互换
        # current_smiles = ut.balance_o_and_s_atoms(current_smiles, self.target_atomNum)

        print('Predicted SMILES is', current_smiles)
        current_element_counts = ut.count_atoms(current_smiles)
        print(f"Predicted Chemical Fomula of final coal model is C{current_element_counts['C']}H{current_element_counts['H']}O{current_element_counts['O']}N{current_element_counts['N']}S{current_element_counts['S']}")
        
        ratio = ut.calculate_unsaturated_carbon_ratio(current_smiles)
        print("Unsatutated carbon rate of final coal model is", ratio)
        
        mass_percentages = ut.calculate_mass_percentages(ut.count_atoms(current_smiles))
        print('Predicted elemental ratio of coal model is:', mass_percentages)
        
        return current_smiles