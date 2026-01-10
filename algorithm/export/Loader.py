import pandas as pd

from algorithm.lattice.Lattice import Lattice
from algorithm.lattice.Voxel import Voxel

class LatticeLoader:
    def __init__(self):
        pass

    @staticmethod
    def create_excel(nx:int, ny:int, nz:int, is_unit_cell: bool=True, 
                     output_path="lattice.xlsx"):
        """Create a template Excel file for inputing an octahedral lattice.
        Args:
            nx, ny, nz: dimensions of the lattice in voxels
            is_unit_cell: whether the lattice is a unit cell (True) or full lattice (False)
            output_path: path to save the generated excel file
        """
        # --- 1. build config sheet ---
        config_df = pd.DataFrame({
            "param": ["nx", "ny", "nz", "is_unit_cell"],
            "value": [nx, ny, nz, bool(is_unit_cell)]
        })

        # --- 2. build lattice sheet ---
        rows = []
        # HEADER ROW: section titles
        header = ["Lattice Coordinates"] + [""] * (nx - 1)
        header += ["", "Cargo Material"] + [""] * (nx - 1)
        header += ["", "Cargo Coordinates"] + [""] * (nx - 1)
        rows.append(header)

        # for each layer (z) from top to bottom
        for z in reversed(range(nz)):
            # layer label row
            layer_row = []
            # lattice coordinates block
            layer_row.extend([f"Layer {z}"] + [""] * (nx - 1))
            layer_row.append("")  
            # cargo material block
            layer_row.extend([f"Layer {z}"] + [""] * (nx - 1))
            layer_row.append("")  
            # cargo coordinates block
            layer_row.extend([f"Layer {z}"] + [""] * (nx - 1))
            rows.append(layer_row)

            # for each y row in this layer (top -> bottom)
            for y in reversed(range(ny)):
                row = []
                for x in range(nx):
                    row.append(f"{x},{y},{z}") # lattice coords
                row.append("") 
                for x in range(nx):
                    row.append(0) # cargo material
                row.append("") 
                for x in range(nx):
                    row.append("0,0,0") # cargo coords
                rows.append(row)

        lattice_df = pd.DataFrame(rows)

        # --- 3. write to excel ---
        with pd.ExcelWriter(f"{output_path}", engine="xlsxwriter") as writer:
            # config sheet: write title in A1, then table below
            config_df.to_excel(writer, sheet_name="Config", startrow=1, index=False)
            cfg_ws = writer.sheets["Config"]
            cfg_ws.write(0, 0, "Lattice Config")

            # Lattice sheet: no header/index
            lattice_df.to_excel(writer, sheet_name="Input", index=False, header=False)

        print(f"Saved lattice template to {output_path}")

    @staticmethod
    def load_excel(filepath="lattice.xlsx") -> Lattice:
        """Read lattice Excel file with sheets 'Config' and 'Lattice',
        returns a MOSES Lattice instance. Assumes the layout produced by create_excel().
        
        Args:
            filepath: path to the excel file
        Returns:
            Lattice: a Lattice instance constructed from the excel file
        """

        # --- 1. read config sheet ---
        config_df = pd.read_excel(filepath, sheet_name="Config", header=1)
        nx = int(config_df.loc[config_df['param'] == 'nx', 'value'].values[0])
        ny = int(config_df.loc[config_df['param'] == 'ny', 'value'].values[0])
        nz = int(config_df.loc[config_df['param'] == 'nz', 'value'].values[0])
        is_unit_cell = bool(config_df.loc[config_df['param'] == 'is_unit_cell', 'value'].values[0])

        # --- 2. read lattice sheet ---
        lattice_df = pd.read_excel(filepath, sheet_name="Lattice", header=None)
        voxels = []
        row_idx = 1
        for z in reversed(range(nz)):
            row_idx += 1  # skip layer label row
            for y in reversed(range(ny)):
                row = lattice_df.iloc[row_idx]
                for x in range(nx):
                    # parse lattice coordinates
                    coord_str = row[x]
                    x_l, y_l, z_l = map(int, coord_str.split(","))
                    voxel_coords = (x_l, y_l, z_l)

                    # parse cargo material
                    cargo = int(row[nx+1 + x])
                    # parse cargo coordinates
                    cargo_coord_str = row[2 * (nx+1) + x]
                    x_c, y_c, z_c = map(int, cargo_coord_str.split(","))
                    cargo_coords = (x_c, y_c, z_c)

                    v = Voxel(
                        coords=voxel_coords,
                        cargo=cargo,
                        cargo_coords=cargo_coords
                    )
                    voxels.append(v)
                
                row_idx += 1

        lattice = Lattice(voxels=voxels, is_unit_cell=is_unit_cell)

        print(f"Loaded lattice from {filepath}")
        return lattice
    
    @staticmethod
    def write_output_sheet(filepath: str, voxels: list[Voxel], sheet_name: str="Output"):
        """
        Append an 'Output' sheet to an existing lattice Excel file.

        Args:
            filepath: path to the existing excel file (with Config and Lattice sheets)
            voxels: list of Voxel objects that form the minimum origami (e.g., moses.mesovoxel.all_voxels).
            sheet_name: name of the sheet to write (default 'Output')
        """
        # map labels -> vertex directions used in Voxel.vertices
        bond_map = {}
        for label, vertex in zip(Voxel.v_names, Voxel.vertices):
            bond_map[label] = vertex

        rows = []
        for _, v in enumerate(voxels):
            row = {
                "Origami": v.id2,
                "Coordinates": f"{v.coords}",
                "Concentration": None, #TODO: later
                "Material": v.cargo
            }
            # fill bond colors for each vertex
            for vertex, bond in v.bonds.items():
                row[bond.get_label()] = bond.color if bond.color else 0
            rows.append(row)

        output_df = pd.DataFrame(
            rows,
            columns=[
                "Origami",
                "Coordinates",
                "Concentration",
                "Material",
                "+x", "-x", "+y", "-y", "+z", "-z",
            ],
        )

        # write into the same excel file, adding/replacing the 'Output' sheet
        with pd.ExcelWriter(
            filepath,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            # startrow=1 to put the title in row 1
            output_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
            ws = writer.sheets[sheet_name]
            ws["A1"] = "Minimum Origami"

        print(f"Appended 'Output' sheet to {filepath}")

    def _write_strand_info(filepath: str, voxels: list[Voxel], vertices_file: str, colors_file: str, sheet_name: str="Strands"):
        """Read in the vertex and color info from the given files to reconstruct the actual
        origami stand order we would need to place given an output of MOSES."""

        # read in vertex info
        vertex_df = pd.read_csv(vertices_file)
        color_df = pd.read_csv(colors_file)

        for v in voxels:
            x, y, z = v.coords
            row = {
                "Origami": v.id2,
                "Coordinates": f"{v.coords}",
                "Concentration": None, #TODO: later
                "Material": v.cargo
            }
            for vertex, bond in v.bonds.items():
                color_row = color_df.loc(color_df['Color'] == abs(bond.color))
                strand = color_row['Strand'] if bond.color>0 else color_row['Complement']
                row["Strand"] = strand

        # combine into one dataframe
        strand_df = vertex_df.merge(color_df, on="Strand ID")

        # write to excel
        with pd.ExcelWriter(
            filepath,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            strand_df.to_excel(writer, sheet_name=sheet_name, index=False)