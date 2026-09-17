"""RDKit descriptor generation: SMILES strings -> the full 1D/2D descriptor matrix."""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from . import config


def compute_rdkit_descriptors(smiles_list) -> pd.DataFrame:
    """All of Descriptors.descList for each SMILES; unparseable SMILES get a row of Nones."""
    names = [name for name, _ in Descriptors.descList]
    calc = MolecularDescriptorCalculator(names)

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([None] * len(names))
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            rows.append([None] * len(names))
            continue
        rows.append(list(calc.CalcDescriptors(mol)))

    return pd.DataFrame(rows, columns=names)


def generate_descriptor_dataset(raw_path=config.RAW_SMILES_FILE,
                                 out_path=config.DESCRIPTORS_FILE) -> pd.DataFrame:
    """Reproduce Case4_1.xlsx from the raw SMILES/EO/PO/Cc dataset."""
    df = pd.read_excel(raw_path)
    desc_df = compute_rdkit_descriptors(df["SMILES"])
    full = pd.concat([df, desc_df], axis=1)
    full.to_excel(out_path, index=False)
    return full


if __name__ == "__main__":
    generate_descriptor_dataset()
    print(f"Wrote descriptors to {config.DESCRIPTORS_FILE}")
