import networkx as nx
import pickle
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as s2m
from rdkit.Chem import MolToSmiles as m2s
from rdkit.Chem import Crippen, Descriptors
from tqdm import tqdm
import re
from helpers import get_ring_structure, get_core_structure, detect_functional_group
import argparse

# Calculate the logP value of a molecule
def logp_cal(mol):
    return Crippen.MolLogP(mol)

# Estimate solubility of a molecule based on logP, molecular weight, and TPSA
def solubility_cal(mol):
    logP = Descriptors.MolLogP(mol)
    mol_weight = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    solubility = -0.01 * logP - 0.005 * mol_weight + 0.03 * tpsa + 1.2
    return solubility

# Define functional groups for hydrogen bond donors and acceptors
hydrogen_bond_donor = [
    'hydroxyl', 'hydroperoxy', 'primary_amine', 'secondary_amine', 
    'hydrazone', 'primary_ketimine', 'secondary_ketimine', 'primary_aldimine',
    'amide', 'sulfhydryl', 'sulfonic_acid', 'thiolester', 'hemiacetal', 
    'hemiketal', 'carboxyl', 'aldoxime', 'ketoxime'
]

hydrogen_bond_acceptor = [
    'ether', 'peroxy', 'haloformyl', 'ketone', 'aldehyde', 'carboxylate', 
    'carboxyl', 'ester', 'ketal', 'carbonate_ester', 'carboxylic_anhydride',
    'primary_amine', 'secondary_amine', 'tertiary_amine', '4_ammonium_ion', 
    'hydrazone', 'primary_ketimine', 'secondary_ketimine', 'primary_aldimine', 
    'amide', 'sulfhydryl', 'sulfonic_acid', 'thiolester', 'aldoxime', 'ketoxime'
]

# Generate node features for each molecule based on SMILES string
def gen_node_feature(sm):
    mol = s2m(sm)
    logp = logp_cal(mol)
    solubility = solubility_cal(mol)

    detect_functional_group(mol)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    atom_types = set()
    bond_types = set()
    is_alkyl = 0
    functional_group = set()
    is_hydrogen_bond_donor, is_hydrogen_bond_acceptor = 0, 0
    
    if mol is not None:
        smiles = sm
        num_rings = ring_info.NumRings()

    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 0:
            atom_types.add(atom.GetSymbol())
        else:
            atom_types.add(f"{atom.GetSymbol()}{atom.GetFormalCharge()}")
        prop = atom.GetProp('FG')
        if prop != '':
            functional_group.add(prop)
    
    if functional_group in hydrogen_bond_donor:
        is_hydrogen_bond_donor = 1
    if functional_group in hydrogen_bond_acceptor:
        is_hydrogen_bond_acceptor = 1

    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            bond_types.add(bond.GetBondTypeAsDouble())

    num_substitutes = smiles.count('*')
    
    # Determine ring structure and core
    if num_rings > 0:
        core_structure = get_ring_structure(mol)
    else:
        core_structure = get_core_structure(mol)
        check = re.sub(r'[CH\-\(\)\[\]/\\@]', '', smiles)
        if len(check) == 0:
            is_alkyl = 1
    core_smiles = m2s(core_structure).upper().replace('-', '')

    return smiles, core_smiles, atom_types, bond_types, functional_group, num_rings, num_substitutes, is_hydrogen_bond_donor, is_hydrogen_bond_acceptor, is_alkyl, logp, solubility

# Create molecular graph and add features
def create_molecular_graph(vocab_path):
    G = nx.DiGraph()

    # Load FG vocabulary:
    with open(vocab_path, 'rb') as f:
        FG_VOCAB = pickle.load(f)

    for sm in tqdm(list(FG_VOCAB)):
        smiles, core_smiles, atom_types, bond_types, functional_group, num_rings, num_substitutes, is_hydrogen_bond_donor, is_hydrogen_bond_acceptor, is_alkyl, logp, solubility = gen_node_feature(sm)
        
        G.add_node(f"smiles={smiles}", label="SMILES")
        G.add_node(f"Core={core_smiles}", label="CoreSMILES")
        G.add_edge(f"smiles={smiles}", f"Core={core_smiles}", label="CoreSMILES")
        
        G.add_node(f"NumRings={num_rings}", label="NumRings")
        G.add_edge(f"smiles={smiles}", f"NumRings={num_rings}", label="NumRings")
        
        G.add_node(f"NumSubstitutes={num_substitutes}", label="NumSubstitutes")
        G.add_edge(f"smiles={smiles}", f"NumSubstitutes={num_substitutes}", label="NumSubstitutes")
        
        G.add_node(f"LogP={round(logp)}", label="LogP")
        G.add_edge(f"smiles={smiles}", f"LogP={round(logp)}", label="LogP")
        
        G.add_node(f"Solubility={round(solubility)}", label="Solubility")
        G.add_edge(f"smiles={smiles}", f"Solubility={round(solubility)}", label="Solubility")
        
        for atom_type in atom_types:
            G.add_node(f"atom_type={atom_type}", label="AtomType")
            G.add_edge(f"smiles={smiles}", f"atom_type={atom_type}", label="Contain_atom")
        
        for bond_type in bond_types:
            G.add_node(f"bond_type={bond_type}", label="BondType")
            G.add_edge(f"smiles={smiles}", f"bond_type={bond_type}", label="Contain_bond")
        
        for fg in functional_group:
            G.add_node(f"FG={fg}", label="FunctionalGroup")
            G.add_edge(f"smiles={smiles}", f"FG={fg}", label='Contain_functional_group')

        if is_hydrogen_bond_donor == 1:
            G.add_node("Contain_hydrogen_bond_donor", label="Contain_hydrogen_bond_donor")
            G.add_edge(f"smiles={smiles}", "Contain_hydrogen_bond_donor", label="Contain_hydrogen_bond_donor")
        
        if is_hydrogen_bond_acceptor == 1:
            G.add_node("Contain_hydrogen_bond_acceptor", label="Contain_hydrogen_bond_acceptor")
            G.add_edge(f"smiles={smiles}", "Contain_hydrogen_bond_acceptor", label="Contain_hydrogen_bond_acceptor")
        
        if is_alkyl == 1:
            G.add_node("Is_alkyl", label="Is_alkyl")
            G.add_edge(f"smiles={smiles}", "Is_alkyl", label="Is_alkyl")

    return G

# Main function to create molecular graph by passing vocab paths
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a FG knowledge graph, process the FG knowledge graph and save data to files.")
    
    # Add arguments
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to the FG vocab file (pkl file).")
    parser.add_argument('--FGKG_path', type=str, required=True, help="Path to save data of the FG knowledge graph (pkl file).")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call create_molecular_graph with the provided paths
    G = create_molecular_graph(vocab_path=args.vocab_path)

    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    relation_to_idx = {relation: i for i, relation in enumerate(set(data['label'] for _, _, data in G.edges(data=True)))}

    head_tensors = []
    relation_tensors = []
    tail_tensors = []

    for u, v, data in G.edges(data=True):
        head_idx = node_to_idx[u]
        relation_idx = relation_to_idx[data['label']]
        tail_idx = node_to_idx[v]
        
        head_tensors.append(torch.tensor(head_idx))
        relation_tensors.append(torch.tensor(relation_idx))
        tail_tensors.append(torch.tensor(tail_idx))

    head_tensor = torch.stack(head_tensors)
    relation_tensor = torch.stack(relation_tensors)
    tail_tensor = torch.stack(tail_tensors)

    head_list = head_tensor.tolist()
    relation_list = relation_tensor.tolist()
    tail_list = tail_tensor.tolist()

    triples = list(zip(head_list, relation_list, tail_list))

    data = {'relation_to_idx': relation_to_idx,
            'node_to_idx': node_to_idx,
            'triples': triples}

    # Save the mappings and triples to files using the paths provided as arguments
    with open(args.FGKG_path, 'wb') as f:
        pickle.dump(data, f)
