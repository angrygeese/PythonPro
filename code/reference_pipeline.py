"""Пайплайн для процессинга молекул.

Читает молекулы из файла, считает дескрипторы и фингерпринты,
делает кластеризацию и визуализирует результаты.
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
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from tmap import TMAP
from types import SimpleNamespace

INPUT_FILE_PATH = os.path.join("..", "data_folder", "tables", "molecules.csv")

OUTPUT_DIR = os.path.join("..", "data_folder", "processed_data")
OUTPUT_FILE_NAME = SimpleNamespace(
    valid="molecules_processed_valid.csv",
    invalid="molecules_processed_invalid.csv"
)
OUTPUT_FILE_PATH = SimpleNamespace(
    valid=os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME.valid),
    invalid=os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME.invalid)
)


MOL_ID = SimpleNamespace(name="mol_id", set_value=lambda mol_id: mol_id)
SMILES = SimpleNamespace(name="smiles", set_value=lambda smiles: smiles)
MOL = SimpleNamespace(name="mol", set_value=lambda data: MolFromSmiles(data[SMILES.name]))
MOL_WEIGHT = SimpleNamespace(name="MW", set_value=lambda data: Descriptors.MolWt(data[MOL.name]))
LOG_P = SimpleNamespace(name="LogP", set_value=lambda data: Descriptors.MolLogP(data[MOL.name]))
TPSA = SimpleNamespace(name="TPSA", set_value=lambda data: Descriptors.TPSA(data[MOL.name]))
HBD = SimpleNamespace(name="HBD", set_value=lambda data: Descriptors.NumHDonors(data[MOL.name]))
HBA = SimpleNamespace(name="HBA", set_value=lambda data: Descriptors.NumHAcceptors(data[MOL.name]))
ROT_BONDS = SimpleNamespace(
    name="RotBonds",
    set_value=lambda data: Descriptors.NumRotatableBonds(data[MOL.name])
)
RINGS = SimpleNamespace(name="Rings", set_value=lambda data: Descriptors.RingCount(data[MOL.name]))
QED = SimpleNamespace(name="QED", set_value=lambda data: Descriptors.qed(data[MOL.name]))
RO5_PASS = SimpleNamespace(
    name="Ro5_pass",
    thresholds={
        MOL_WEIGHT.name: SimpleNamespace(value=500, sign=operator.le),
        LOG_P.name: SimpleNamespace(value=5, sign=operator.le),
        HBD.name: SimpleNamespace(value=5, sign=operator.le),
        HBA.name: SimpleNamespace(value=10, sign=operator.le)
    },
    set_value=lambda thresholds, data: all(
        [field_name.sign(data[field_name], field_name.value) for field_name in thresholds]
    )
)

mw_list = []
logp_list = []
tpsa_list = []
hbd_list = []
hba_list = []
rotbonds_list = []
rings_list = []
qed_list = []
valid_list = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE_PATH, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
# process all molecules fast, don't touch
mols = [MOL.set_value(r) for r in rows]
for r in rows:
    mol = MOL.set_value(r)
    print("ok")
    if mol is not None:
        r[MOL.name] = mol
        mw_list.append(MOL_WEIGHT.set_value(r))
        logp_list.append(LOG_P.set_value(r))
        tpsa_list.append(TPSA.set_value(r))
        hbd_list.append(HBD.set_value(r))
        hba_list.append(HBA.set_value(r))
        rotbonds_list.append(ROT_BONDS.set_value(r))
        rings_list.append(RINGS.set_value(r))
        qed_list.append(QED.set_value(r))
        valid_list.append(True)
    else:
        mw_list.append(None)
        logp_list.append(None)
        tpsa_list.append(None)
        hbd_list.append(None)
        hba_list.append(None)
        rotbonds_list.append(None)
        rings_list.append(None)
        qed_list.append(None)
        valid_list.append(False)

for i, r in enumerate(rows):
    r["valid"] = valid_list[i]
    r["MW"] = mw_list[i]
    r['LogP'] = logp_list[i]
    r['TPSA'] = tpsa_list[i]
    r["HBD"] = hbd_list[i]
    r["HBA"] = hba_list[i]
    r["RotBonds"] = rotbonds_list[i]
    r["Rings"] = rings_list[i]
    r["QED"] = qed_list[i]
    r["Ro5_pass"] = (r["MW"] <= 500) and (r["LogP"] <= 5) and (r["HBD"] <= 5) and (r["HBA"] <= 10)
# print statistics
print(f"Valid molecules: {sum(valid_list)} / {len(rows)}")
print(f"Ro5 pass: {sum(r['Ro5_pass'] for r in rows)} / {len(rows)}")
print(f"MW range: {min(mw_list):.1f} — {max(mw_list):.1f}")
for r in rows[:10]:
    print(r["mol_id"], r["MW"], r["LogP"], r["TPSA"], r["QED"], r["Ro5_pass"])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
fps = [gen.GetFingerprint(m) for m in mols]
distances = []
for i in range(1, len(fps)):
    sims = BulkTanimotoSimilarity(fps[i], fps[:i])
    distances.extend([1 - s for s in sims])
clusters = Butina.ClusterData(distances, len(fps), 0.7, isDistData=True)
for r in rows:
    r['cluster'] = -1
for cluster_id, members in enumerate(clusters):
    for idx in members:
        rows[idx]['cluster'] = cluster_id
X = np.array([list(fp) for fp in fps], dtype=np.uint8)
x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(X)
for r, xi, yi in zip(rows, x, y):
    r["x"] = xi
    r["y"] = yi
with open(OUTPUT_FILE_PATH.valid, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
labels = [f"{r['smiles']}__{r['mol_id']}" for r in rows]
# use Faerun
f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
f.add_scatter(
    "molecules", { "x": x, "y": y, "c": [[r["cluster"] for r in rows], [r["LogP"] for r in rows], [r["MW"] for r in rows]], "labels": labels},
    shader="smoothCircle", point_scale=10, colormap=["tab20", "viridis", "plasma"], categorical=[True, False, False],
    series_title=["Cluster", "LogP", "MolWt"], has_legend=True, legend_title=["Cluster", "LogP", "MolWt"]
)
f.add_tree(
    "molecules_tree", {"from": s, "to": t}, point_helper="molecules", color="#222222",
)
f.plot("molecules", path=OUTPUT_DIR, template="smiles")
# check the values in the browser
