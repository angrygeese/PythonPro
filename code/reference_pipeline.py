"""
Pipeline for molecule processing.

Identifies molecules using SMILES notation, computes specified proterties 
into fields of the table, performs clasterization and plots the result.
"""

import csv
import operator
import os

import numpy as np

from faerun import Faerun
from rdkit.Chem import (
    Crippen,
    Descriptors,
    MolFromSmiles,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from tmap import TMAP
from types import SimpleNamespace  # используется вместо словарей,
                                   # т. к. позволяет использовать точечную
                                   # нотацию для получения значения
                                   # (SimpleNamespace.attr вместо dict["key"])

REQUIRED_FIELDS = SimpleNamespace(
    MOL_ID=SimpleNamespace(
        name="mol_id"
    ),
    SMILES=SimpleNamespace(
        name="smiles"
    )
)

CALCULATED_FIELDS = SimpleNamespace(
    FINGERPRINT=SimpleNamespace(
        name="fingerprint",
        calculate_value=(rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=4096
        ).GetFingerprint)
    ),
    MOL_WEIGHT=SimpleNamespace(
        name="MW",
        calculate_value=Descriptors.MolWt
    ),
    LOG_P=SimpleNamespace(
        name="LogP",
        calculate_value=Descriptors.MolLogP
    ),
    TPSA = SimpleNamespace(
        name="TPSA",
        calculate_value=Descriptors.TPSA
    ),
    HBD = SimpleNamespace(
        name="HBD",
        calculate_value=Descriptors.NumHDonors
    ),
    HBA = SimpleNamespace(
        name="HBA",
        calculate_value=Descriptors.NumHAcceptors
    ),
    ROT_BONDS = SimpleNamespace(
        name="RotBonds",
        calculate_value=Descriptors.NumRotatableBonds
    ),
    RINGS = SimpleNamespace(
        name="Rings",
        calculate_value=Descriptors.RingCount
    ),
    QED = SimpleNamespace(
        name="QED",
        calculate_value=Descriptors.qed
    ),
    RO5_PASS = SimpleNamespace(
        name="Ro5_pass",
        thresholds=None,
        calculate_value=None
    ),
    CLUSTER = SimpleNamespace(
        name="cluster",
        calculate_value=(lambda value: None if isinstance(value, Mol) else value)
    ),
    X = SimpleNamespace(
        name="x",
        calculate_value=(lambda value: None if isinstance(value, Mol) else value)
    ),
    Y = SimpleNamespace(
        name="y",
        calculate_value=(lambda value: None if isinstance(value, Mol) else value)
    )
)


CALCULATED_FIELDS.RO5_PASS.thresholds = (
    SimpleNamespace(
        field=CALCULATED_FIELDS.MOL_WEIGHT,
        value=500,
        sign=operator.le
    ),
    SimpleNamespace(
        field=CALCULATED_FIELDS.LOG_P,
        value=5,
        sign=operator.le
    ),
    SimpleNamespace(
        field=CALCULATED_FIELDS.HBD,
        value=5,
        sign=operator.le
    ),
    SimpleNamespace(
        field=CALCULATED_FIELDS.HBA,
        value=10,
        sign=operator.le
    )
)

CALCULATED_FIELDS.RO5_PASS.calculate_value = lambda molecule: all(
    [
        threshold.sign(
            threshold.field.calculate_value(molecule), threshold.value
        ) for threshold in CALCULATED_FIELDS.RO5_PASS.thresholds
    ]
)

INPUT = SimpleNamespace(
    directory=SimpleNamespace(
        name=os.path.join("..", "data_folder", "tables")
    ),
    file=SimpleNamespace(
        name="molecules.csv",
        fields=(
            REQUIRED_FIELDS.MOL_ID,
            REQUIRED_FIELDS.SMILES
        ),
    )
)

OUTPUT = SimpleNamespace(
    statistics=SimpleNamespace(
        length=10,
        fields=(
            REQUIRED_FIELDS.MOL_ID,
            CALCULATED_FIELDS.MOL_WEIGHT,
            CALCULATED_FIELDS.LOG_P,
            CALCULATED_FIELDS.TPSA,
            CALCULATED_FIELDS.QED,
            CALCULATED_FIELDS.RO5_PASS
        )
    ),
    plot=SimpleNamespace(
        name="molecules",
        fields=(
            CALCULATED_FIELDS.CLUSTER,
            CALCULATED_FIELDS.LOG_P,
            CALCULATED_FIELDS.MOL_WEIGHT
        )
    ),
    directory=SimpleNamespace(
        name=os.path.join("..", "data_folder", "processed_data")
    ),
    file=SimpleNamespace(
        name = SimpleNamespace(
            valid="molecules_processed_valid.csv",
            invalid="molecules_processed_invalid.csv"
        ),
        fields=(
            REQUIRED_FIELDS.MOL_ID,
            REQUIRED_FIELDS.SMILES,
            CALCULATED_FIELDS.MOL_WEIGHT,
            CALCULATED_FIELDS.LOG_P,
            CALCULATED_FIELDS.TPSA,
            CALCULATED_FIELDS.HBD,
            CALCULATED_FIELDS.HBA,
            CALCULATED_FIELDS.ROT_BONDS,
            CALCULATED_FIELDS.RINGS,
            CALCULATED_FIELDS.QED,
            CALCULATED_FIELDS.RO5_PASS
        )
    ),
)

def check_header(header: list[str]) -> bool:
    required_fields = [field.name for field in vars(REQUIRED_FIELDS).values()]
    return set(required_fields).issubset(set(header))

def update_table(input_file_path: str) -> tuple[list[dict], list[dict]]:
    """
    Computes the specified properties for each molecule from source table
    and stores results into dictionary.
    """
    with open(input_file_path, newline="", encoding="utf-8") as input_file:
        header = input_file.readline().rstrip().split(",")
        if check_header(header):
            molecules_valid, molecules_invalid = [], []
            for line in input_file:
                row_values = dict(zip(header, line.rstrip().split(",")))
                molecule = MolFromSmiles(row_values[REQUIRED_FIELDS.SMILES.name])
                calculated_fields = vars(CALCULATED_FIELDS).values()
                if molecule:
                    for field in calculated_fields:
                        row_values.update(
                            {field.name: field.calculate_value(molecule)}
                        )
                    molecules_valid.append(row_values)
                else:
                    molecules_invalid.append(row_values)
        else:
            raise KeyError("Error reading source table: invalid header.")
    return molecules_valid, molecules_invalid

def print_statistics(molecules_valid: list, molecules_invalid: list) -> None:
    """
    Prints some statistical info.
    """
    total_count = len(molecules_valid) + len(molecules_invalid)
    print(f"Valid molecules: {len(molecules_valid)} / {total_count}")
    print(f"Ro5 passed: {sum(molecule[CALCULATED_FIELDS.RO5_PASS.name] for molecule in molecules_valid)} / {total_count}")
    molecular_weights = [molecule[CALCULATED_FIELDS.MOL_WEIGHT.name] for molecule in molecules_valid]
    print(f"Molecular weights range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
    print()
    print(*[field.name for field in OUTPUT.statistics.fields], sep="\t")
    for molecule in molecules_valid[:OUTPUT.statistics.length]:
        print(*[molecule[field.name] for field in OUTPUT.statistics.fields], sep="\t")
    print()

def clusterize(molecules_valid: list[dict]) -> None:
    """
    Performs clasterization.
    """
    fingerprints = [
        molecule[CALCULATED_FIELDS.FINGERPRINT.name] for molecule in molecules_valid
    ]
    distances = []
    for index in range(1, len(fingerprints)):
        similarity_scores = BulkTanimotoSimilarity(
            fingerprints[index],
            fingerprints[:index]
        )
        distances.extend([
            1 - similarity_score for similarity_score in similarity_scores
        ])
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        0.7,
        isDistData=True
    )
    for cluster_id, members in enumerate(clusters):
        for index in members:
            molecules_valid[index][CALCULATED_FIELDS.CLUSTER.name] = cluster_id

def plot_graph(
        molecules_valid: list[dict],
        output_dir: str = OUTPUT.directory.name,
        plot_name: str = OUTPUT.plot.name
) -> None:
    """
    Creates layout and draws a graph using Faerun library.
    """
    fingerprints = np.array(
        [
            list(molecule[CALCULATED_FIELDS.FINGERPRINT.name]) for molecule in molecules_valid
        ], dtype=np.uint8
    )
    x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(fingerprints)
    for molecule, xi, yi in zip(molecules_valid, x, y):
        molecule[CALCULATED_FIELDS.X.name] = xi
        molecule[CALCULATED_FIELDS.Y.name] = yi

    f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
    
    plot_fields_names = [field.name for field in OUTPUT.plot.fields]
    required_fields_names = [
        field.name for field in vars(REQUIRED_FIELDS).values()
    ]
    f.add_scatter(
        name=plot_name, 
        data={
            "x": x,
            "y": y,
            "c": [
                [row[field_name] for row in molecules_valid] for field_name in plot_fields_names
            ], 
            "labels": [
                f"{'__'.join([row[field_name] for field_name in required_fields_names[::-1]])}" for row in molecules_valid
            ]
        },
        shader="smoothCircle",
        point_scale=10, colormap=["tab20", "viridis", "plasma"],
        categorical=[True, False, False],
        series_title=plot_fields_names,
        has_legend=True,
        legend_title=plot_fields_names
    )
    f.add_tree(
        name=f"{plot_name}_tree",
        data={"from": s, "to": t},
        point_helper=plot_name,
        color="#222222",
    )
    f.plot(plot_name, path=output_dir, template=REQUIRED_FIELDS.SMILES.name)

def filter_output_fields(molecules: list[dict]) -> list[dict]:
    """
    Filters record info only to required fields.
    """
    return [
        {field.name: molecule[field.name] for field in OUTPUT.file.fields}
        for molecule in molecules
    ]

def save(
        values: list[dict],
        file_path: str
) -> None:
    """
    Saves data to a specified file.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[field.name for field in OUTPUT.file.fields]
        )
        writer.writeheader()
        writer.writerows(values)
        print(f"{file_path}\tsaved")

def run() -> None:
    """
    Orchestrates the pipeline.
    """
    input_path = os.path.join(INPUT.directory.name, INPUT.file.name)
    output_path_valid = os.path.join(OUTPUT.directory.name, OUTPUT.file.name.valid)
    output_path_invalid = os.path.join(OUTPUT.directory.name, OUTPUT.file.name.invalid)

    molecules_valid, molecules_invalid = update_table(input_path)
    if any([molecules_valid, molecules_invalid]):
        os.makedirs(OUTPUT.directory.name, exist_ok=True)
        if molecules_valid:
            print_statistics(molecules_valid, molecules_invalid)
            clusterize(molecules_valid)
            plot_graph(molecules_valid)
            molecules_valid_filtered = filter_output_fields(molecules_valid)
            molecules_invalid_filtered = filter_output_fields(molecules_invalid)
            save(molecules_valid_filtered, output_path_valid)
        if molecules_invalid:
            save(molecules_invalid_filtered, output_path_invalid)


if __name__ == "__main__":
    run()
