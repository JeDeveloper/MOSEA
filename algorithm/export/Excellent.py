from dataclasses import dataclass
import pandas as pd
import os
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from algorithm.lattice.Lattice import Lattice
from algorithm.lattice.Voxel import Voxel
from algorithm.symmetry.SymmetryDf import SymmetryDf
from algorithm.painting.Mesovoxel import Mesovoxel

@dataclass
class LatticeSheet:
    sheet_name: str = "Lattice"
    title: str = "Unit Cell Lattice"

    columns: tuple[str, ...] = (
        "Voxel",
        "Mesovoxel_ID",
        "Coords",
        "Cargo",
        "Cargo_Coords",
        "x+", "x-",
        "y+", "y-",
        "z+", "z-",
    )

class Excellent:
    def __init__(self):
        pass

    @staticmethod
    def init_excel(nx: int, ny: int, nz: int, is_unit_cell: bool=True,
                   output_path="lattice.xlsx"):
        """Initialize a template Excel file for inputing an octahedral lattice.
        
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

        # --- 2. build lattice sheet, row-by-row ---
        rows = []
        # HEADER ROW: section titles
        header = ["Lattice Coordinates"] + [""]*(nx - 1)
        header += ["", "Cargo Material"] + [""]*(nx - 1)
        header += ["", "Cargo Coordinates"] + [""]*(nx - 1)
        rows.append(header)

        # for each layer (z) from top to bottom
        # recall we are building this row by row
        for z in reversed(range(nz)):
            # LAYER_LABEL ROW
            layer_row = []
            # lattice coordinates block
            layer_row.extend([f"Layer {z}"] + [""]*(nx - 1))
            layer_row.append("")  
            # cargo material block
            layer_row.extend([f"Layer {z}"] + [""]*(nx - 1))
            layer_row.append("")  
            # cargo coordinates block
            layer_row.extend([f"Layer {z}"] + [""]*(nx - 1))
            rows.append(layer_row)
    
            # for each y row in this layer (top -> bottom)
            for y in reversed(range(ny)):
                row = []
                # --> lattice coords block
                for x in range(nx):
                    row.append(f"{x},{y},{z}") # (user doesn't touch, ideally)
                row.append("") 
                # --> cargo material block
                for x in range(nx): 
                    row.append(0) # user touches
                row.append("") 
                # --> cargo coords block
                for x in range(nx):
                    row.append("0,0,0") # user can also touch if they please
                rows.append(row)

        lattice_df = pd.DataFrame(rows)

        # --- 3. write to excel ---
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # config sheet
            config_df.to_excel(writer, sheet_name="Config", startrow=1, index=False)
            cfg_ws = writer.sheets["Config"]
            cfg_ws["A1"] = "Lattice Config"

            # lattice sheet: no header/index
            lattice_df.to_excel(writer, sheet_name="Input", index=False, header=False)

        print(f"Saved lattice template to {output_path}")

    @staticmethod
    def load_input(filepath: str="lattice.xlsx") -> Lattice:
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
        lattice_df = pd.read_excel(filepath, sheet_name="Input", header=None)
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

        voxels.sort(key=lambda v: (v.coords[2], v.coords[1], v.coords[0]))
        # for i, v in enumerate(voxels):
        #     print(f"Loaded Voxel {i} at {v.coords} with cargo {v.cargo} @ {v.cargo_coords}")
        lattice = Lattice(voxels=voxels, is_unit_cell=is_unit_cell)

        print(f"Loaded input lattice from {filepath}")
        return lattice

    @staticmethod
    def _voxel_to_row(v: Voxel) -> dict:
        """convert a voxel to a dict representing a row in the excel sheet"""
        get_bond = lambda vertex: v.get_bond(vertex).color if v.get_bond(vertex).color else 0

        row = {
            "Voxel": v.id,
            "Mesovoxel_ID": v.id2,
            "Coords": f"{v.coords}",
            "Cargo": v.cargo,
            "Cargo_Coords": f"{v.cargo_coords}",
            "x+": get_bond((0.5, 0, 0)),
            "x-": get_bond((-0.5, 0, 0)),
            "y+": get_bond((0, 0.5, 0)),
            "y-": get_bond((0, -0.5, 0)),
            "z+": get_bond((0, 0, 0.5)),
            "z-": get_bond((0, 0, -0.5)),
        }
        return row

    @staticmethod
    def _mvoxel_to_row(mv: Voxel) -> dict:
        """convert a mvoxel to a dict representing a row in the excel sheet"""
        get_bond = lambda vertex: mv.get_bond(vertex).color if mv.get_bond(vertex).color else 0

        row = {
            "Mesovoxel_ID": mv.id2,
            "Coords": f"{mv.coords}",
            "Concentration": None, # placeholder for later
            "Cargo": mv.cargo,
            "Cargo_Coords": f"{mv.cargo_coords}",
            "x+": get_bond((0.5, 0, 0)),
            "x-": get_bond((-0.5, 0, 0)),
            "y+": get_bond((0, 0.5, 0)),
            "y-": get_bond((0, -0.5, 0)),
            "z+": get_bond((0, 0, 0.5)),
            "z-": get_bond((0, 0, -0.5)),
        }
        return row
    
    @staticmethod
    def write_output(filepath: str, moses_obj):
        # write the following to their own sheets in the given filepath
        Excellent.write_lattice(filepath, moses_obj.lattice)
        Excellent.write_symmetry_df(filepath, moses_obj.symmetry_df)
        Excellent.write_mesovoxel(filepath, moses_obj.mesovoxel)
    
    @staticmethod
    def write_lattice(filepath: str, lattice: Lattice):
        """
        Write the lattice sheet to an excel file.

        Args:
            filepath: path to the excel file to write to
            lattice: Lattice object containing voxels to write
        """
        # just do df -> excel
        rows = [Excellent._voxel_to_row(v) for v in lattice.voxels + lattice.unit_cell_voxels]
        lattice_df = pd.DataFrame(rows, columns=LatticeSheet.columns)

        if not os.path.exists(filepath):
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # create empty workbook
                pd.DataFrame().to_excel(writer, sheet_name="__init__", index=False)

        # write df starting at row 3 (leave space for title)
        with pd.ExcelWriter(
            filepath,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            lattice_df.to_excel(
                writer, 
                sheet_name=LatticeSheet.sheet_name, 
                index=False, 
                header=True,
                startrow=0
            )

        print(f"Wrote lattice sheet to {filepath}")

    @staticmethod
    def write_symmetry_df(filepath: str, symmetry_df: SymmetryDf):
        """
        Write the symmetry dataframe to an excel file.

        Args:
            filepath: path to the excel file to write to
            symmetry_df: SymmetryDf object containing the symmetry dataframe
        """
        with pd.ExcelWriter(
            filepath,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            symmetry_df.symmetry_df.to_excel(
                writer, 
                sheet_name="Symmetry", 
                index=True, 
                header=True,
                startrow=1
            )
            ws = writer.sheets["Symmetry"]
            ws["A1"] = "Lattice Symmetries"
            ws.freeze_panes = "B3" 

        print(f"Wrote symmetry_df to {filepath}")

    @staticmethod
    def write_mesovoxel(filepath: str, mesovoxel: Mesovoxel):
        """
        Write the mesovoxel sheet to an excel file.

        Args:
            filepath: path to the excel file to write to
            mesovoxel: Mesovoxel object containing the mesovoxels to write
        """
        mvoxels = [mesovoxel.lattice.get_voxel(mv_id) for mv_id in mesovoxel.all_voxels()]
        rows = [Excellent._mvoxel_to_row(mv) for mv in mvoxels]
        meso_df = pd.DataFrame(
            rows,
            columns=[
                "Mesovoxel_ID",
                "Coords",
                "Concentration",
                "Cargo",
                "Cargo_Coords",
                "x+", "x-",
                "y+", "y-",
                "z+", "z-",
            ],
        )

        with pd.ExcelWriter(
            filepath,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            meso_df.to_excel(
                writer, 
                sheet_name="Mesovoxel", 
                index=False, 
                header=True,
                startrow=0
            )
            # meso_ws = writer.sheets["Mesovoxel"]
            # meso_ws.write(0, 0, "Mesovoxel")

        print(f"Wrote mesovoxel sheet to {filepath}")

    @staticmethod
    def read_output(filepath: str):
        """Read the output lattice from a file that's been through
        the MOSES x Excellent pipeline ;)"""
        config_df = pd.read_excel(filepath, sheet_name="Config", header=1)
        lat_df = pd.read_excel(filepath, sheet_name="Lattice", header=0)
        meso_df = pd.read_excel(filepath, sheet_name="Mesovoxel", header=0)

        cfg = config_df.set_index("param")["value"]
        xdim = int(cfg["nx"])
        ydim = int(cfg["ny"])
        zdim = int(cfg["nz"])
        is_unit_cell = bool(cfg["is_unit_cell"])

        if is_unit_cell:
            xdim, ydim, zdim = xdim-1, ydim-1, zdim-1
            
        n_voxels = xdim*ydim*zdim # anything beyond this is a unit cell voxel

        lat_voxels = []
        unit_voxels = []
        for _, r in lat_df.iterrows():
            v = Voxel(
                coords=tuple(map(int, r["Coords"].strip("()").split(","))),
                cargo=int(r["Cargo"]),
                cargo_coords=tuple(map(int, r["Cargo_Coords"].strip("()").split(",")))
            )
            v.id = int(r["Voxel"])
            v.set_id2(int(r["Mesovoxel_ID"]))
            # set bonds
            for vertex, label in zip(
                v.vertices,
                ["x+", "x-", "y+", "y-", "z+", "z-"]
            ):
                color = r[label]
                if pd.notna(color) and color != 0:
                    v.get_bond(vertex).set_color(int(color))
            if v.id < n_voxels:
                lat_voxels.append(v)
            else:
                unit_voxels.append(v)

        meso_voxels = []
        for _, r in meso_df.iterrows():
            mv = Voxel(
                coords=tuple(map(int, r["Coords"].strip("()").split(","))),
                cargo=int(r["Cargo"]),
                cargo_coords=tuple(map(int, r["Cargo_Coords"].strip("()").split(",")))
            )
            mv.set_id2(int(r["Mesovoxel_ID"]))
            # set bonds
            for vertex, label in zip(
                mv.vertices,
                ["x+", "x-", "y+", "y-", "z+", "z-"]
            ):
                color = r[label]
                if pd.notna(color) and color != 0:
                    mv.get_bond(vertex).set_color(int(color))
            meso_voxels.append(mv)

        # now just return those 3 voxel lists, to be used in visualizer
        return lat_voxels, unit_voxels, meso_voxels