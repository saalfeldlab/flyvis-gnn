"""
Fetch FlyWire neuron meshes for the 65 cell types used in flyvis-gnn.
Run this script on a machine with internet access.

Usage:
    pip install caveclient fafbseg navis cloudvolume
    python fetch_flywire_meshes.py

Output:
    flywire_meshes/  directory with one .obj file per cell type
    flywire_meshes/root_id_map.json  mapping cell_type -> root_id
"""

import json
import os
import numpy as np

# ---- Configuration ----
CAVE_TOKEN = "4fb39234d1c8648e89200cd374c1a636"
OUTPUT_DIR = "flywire_meshes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 65 cell types used in flyvis-gnn (sorted alphabetically)
CELL_TYPES = [
    'Am', 'C2', 'C3', 'CT1(Lo1)', 'CT1(M10)',
    'L1', 'L2', 'L3', 'L4', 'L5', 'Lawf1', 'Lawf2',
    'Mi1', 'Mi10', 'Mi11', 'Mi12', 'Mi13', 'Mi14', 'Mi15',
    'Mi2', 'Mi3', 'Mi4', 'Mi9',
    'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
    'T1', 'T2', 'T2a', 'T3', 'T4a', 'T4b', 'T4c', 'T4d',
    'T5a', 'T5b', 'T5c', 'T5d',
    'Tm1', 'Tm16', 'Tm2', 'Tm20', 'Tm28', 'Tm3', 'Tm30', 'Tm4',
    'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c', 'Tm9',
    'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18',
    'TmY3', 'TmY4', 'TmY5a', 'TmY9',
]

# ---- Step 1: Get root_ids from CAVE ----
print("Step 1: Connecting to CAVE...")
import caveclient
client = caveclient.CAVEclient('flywire_fafb_production')
client.auth.token = CAVE_TOKEN

print("Step 2: Querying cell type annotations...")
# Query the cell_type annotation table for our types
# Try common table names
try:
    tables = client.materialize.get_tables()
    print(f"  Available tables ({len(tables)}): {tables[:10]}...")

    # Look for cell type / classification table
    type_tables = [t for t in tables if any(kw in t.lower() for kw in
                   ['cell_type', 'classification', 'neuron_type', 'optic'])]
    print(f"  Candidate tables: {type_tables}")
except Exception as e:
    print(f"  Error listing tables: {e}")
    type_tables = []

# Try the most common table names
for table_name in type_tables + ['classification_system', 'cell_type_local',
                                   'neuron_information_v2', 'proofreading_status_public_v1']:
    try:
        print(f"  Trying table: {table_name}")
        meta = client.materialize.get_table_metadata(table_name)
        print(f"    Schema: {meta}")
        # Query a small sample
        sample = client.materialize.query_table(table_name, limit=5)
        print(f"    Columns: {list(sample.columns)}")
        print(f"    Sample:\n{sample.head(2)}")
        break
    except Exception as e:
        print(f"    Failed: {e}")

# ---- Step 2: For each cell type, get one root_id ----
print("\nStep 3: Getting one root_id per cell type...")
root_id_map = {}
failed = []

for ct in CELL_TYPES:
    try:
        # Try querying with cell type name
        # Adjust the table_name and column based on what Step 2 found
        df = client.materialize.query_table(
            table_name,  # use whichever table worked above
            filter_equal_dict={'cell_type': ct},  # adjust column name if needed
            limit=1
        )
        if len(df) > 0:
            # root_id column might be 'pt_root_id' or 'root_id'
            rid_col = [c for c in df.columns if 'root_id' in c.lower()][0]
            root_id_map[ct] = int(df[rid_col].iloc[0])
            print(f"  {ct}: root_id = {root_id_map[ct]}")
        else:
            print(f"  {ct}: NO RESULTS")
            failed.append(ct)
    except Exception as e:
        print(f"  {ct}: ERROR - {e}")
        failed.append(ct)

print(f"\nFound {len(root_id_map)}/{len(CELL_TYPES)} root_ids")
if failed:
    print(f"Failed: {failed}")

# Save mapping
with open(os.path.join(OUTPUT_DIR, "root_id_map.json"), 'w') as f:
    json.dump(root_id_map, f, indent=2)
print(f"Saved root_id_map.json")

# ---- Step 3: Download meshes ----
print("\nStep 4: Downloading meshes...")
import cloudvolume

# FlyWire segmentation volume
vol = cloudvolume.CloudVolume(
    'precomputed://gs://flywire_v141_m783',
    use_https=True,
    progress=False,
)

for ct, rid in root_id_map.items():
    obj_path = os.path.join(OUTPUT_DIR, f"{ct}.obj")
    if os.path.exists(obj_path):
        print(f"  {ct}: already exists, skipping")
        continue
    try:
        mesh = vol.mesh.get(rid)[rid]
        # Save as OBJ
        with open(obj_path, 'w') as f:
            for v in mesh.vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in mesh.faces.reshape(-1, 3):
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces) // 3
        print(f"  {ct}: {n_verts} vertices, {n_faces} faces -> {obj_path}")
    except Exception as e:
        print(f"  {ct}: MESH ERROR - {e}")
        # Try alternative: fafbseg
        try:
            import fafbseg
            mesh = fafbseg.flywire.get_mesh_neuron(rid)
            mesh.export(obj_path)
            print(f"  {ct}: got via fafbseg -> {obj_path}")
        except Exception as e2:
            print(f"  {ct}: fafbseg also failed - {e2}")

print("\n=== DONE ===")
print(f"Meshes saved to {OUTPUT_DIR}/")
print(f"Copy this directory into the devcontainer:")
print(f"  docker cp {OUTPUT_DIR} <container>:/workspace/flyvis-gnn/flywire_meshes")
