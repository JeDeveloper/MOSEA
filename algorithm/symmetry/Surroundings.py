import numpy as np
from algorithm.lattice.Lattice import Lattice

class Surroundings:
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        xdim, ydim, zdim = lattice.dimensions
        self.max_dim = max(xdim, ydim, zdim)

        # note: rel is a cube of coordinates centered at (0,0,0)
        r = np.arange(-self.max_dim, self.max_dim + 1, dtype=np.int32)
        x, y, z = np.meshgrid(r, r, r, indexing="ij")
        self.rel = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # (K, 3)

    def voxel_surroundings(self, voxel) -> dict[tuple[float, float, float], int]:
        """
        create a cube of surrounding particles all oriented wrt. where 
        v.cargo_coords would be
        """
        v = self.lattice.get_voxel(voxel)

        xdim, ydim, zdim = self.lattice.dimensions
        max_dim = max(xdim, ydim, zdim)

        # create a surroundings cube at least max_len out from the center
        coord_range = np.linspace(-max_dim, max_dim, 2*max_dim+1)
        x, y, z = np.meshgrid(coord_range, coord_range, coord_range, indexing='ij')
        coords = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        surr = {}
        for coord in coords:
            # convert voxel_surr coords into an index into the original lattice
            x, y, z = coord + np.array(v.coords)
            og_x = int((xdim + x) % xdim)
            og_y = int((ydim + y) % ydim)
            og_z = int((zdim + z) % zdim)
            
            og_voxel = self.lattice.get_voxel((og_x, og_y, og_z))

            # translate the coordinate a little for the accurate surroundings
            coord = coord + np.array(og_voxel.cargo_coords)
            surr[tuple(coord)] = og_voxel.cargo

        return surr
    
    def voxel_surroundings2(self, voxel) -> tuple[list[tuple[float, float, float]], list[int]]:
        """
        Returns a cube of the surrounding cargo coordinates with respect to the supplied one.
        The dimensions are (2*max_dim+1)^3, where max_dim is the largest dimension of the lattice.
        The returned lists are ordered such that coords[i] corresponds to cargos[i].

        Args:
            voxel: Voxel or voxel.id corresponding to what you want the surroundings of
        Returns:
            coords: List of coordinates (x, y, z) of the surroundings with respect to
                    the supplied voxel's cargo position
            cargos: List of cargo types at each corresponding coordinate
        """
        v = self.lattice.get_voxel(voxel)
        rel = self.rel

        # get coordinates into our original 'absolute' lattice bounds
        abs_xyz = rel + np.array(v.coords, dtype=np.int32)
        abs_x = abs_xyz[:,0] % self.lattice.xdim
        abs_y = abs_xyz[:,1] % self.lattice.ydim
        abs_z = abs_xyz[:,2] % self.lattice.zdim

        # collect coordinates and cargos
        K = rel.shape[0]
        coords = np.empty((K, 3), dtype=np.float32)
        cargos = np.empty((K,), dtype=np.int32)
        for i in range(K):
            # we get the original voxel from the absolute coordinate system
            og_voxel = self.lattice.get_voxel((int(abs_x[i]), int(abs_y[i]), int(abs_z[i])))

            # but construct the surroundings from our relative (translated) one 
            coords[i, 0] = float(rel[i, 0]) + float(og_voxel.cargo_coords[0])
            coords[i, 1] = float(rel[i, 1]) + float(og_voxel.cargo_coords[1])
            coords[i, 2] = float(rel[i, 2]) + float(og_voxel.cargo_coords[2])
            cargos[i] = og_voxel.cargo

        return coords, cargos

    def rotate(self, surr_dict: dict[tuple[float, float, float], int], rotation) -> dict[tuple[float, float, float], int]:
        """
        accepts a surroundings dictionary (coords: cargo) and rotates each coordinate 
        based on the supplied rotation function
        """
        # convert coords and materials to their own np.arrays
        surr_keys = np.array(list(surr_dict.keys()))
        surr_values = list(surr_dict.values())

        # apply rotation
        rot_surr_keys = rotation(surr_keys)
        rot_surr_keys = np.round(rot_surr_keys, 2)
        rot_surr = {tuple(key): value for key, value in zip(rot_surr_keys, surr_values)}

        return rot_surr
    
    def rotate2(self, coords: np.ndarray, rotation, round_decimals=2) -> np.ndarray:
        """
        accepts a surroundings coordinates array (N, 3) and rotates each coordinate 
        based on the supplied rotation function
        """
        rot_coords = rotation(coords)
        rot_coords = np.round(rot_coords, round_decimals)
        return rot_coords.astype(np.float32)

    def canonicalize(self, coords: np.ndarray, cargos: np.ndarray, decimals: int = 2):
        """
        Turn (coords, cargos) into a canonical order so we can compare with array_equal.
        - rounds coords to kill float noise
        - sorts rows lexicographically by (x,y,z)
        """
        # c = np.round(coords, decimals=decimals)

        # lexsort wants keys last-to-first, so z, then y, then x
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))

        c_sorted = coords[order]
        cargos_sorted = cargos[order]
        return c_sorted, cargos_sorted

if __name__ == "__main__":
    from algorithm.lattice.Voxel import Voxel

    # a sample oriented lattice
    # --- layer 0 ---
    v0 = Voxel(coords=(0,0,0), cargo=1, cargo_coords=(0,0,0))
    v1 = Voxel(coords=(1,0,0), cargo=1, cargo_coords=(0,0,-0.5))
    v2 = Voxel(coords=(0,1,0), cargo=1, cargo_coords=(0,0,-0.5))
    v3 = Voxel(coords=(1,1,0), cargo=1, cargo_coords=(0,0,0))

    # --- layer 1 ---
    v4 = Voxel(coords=(0,0,1), cargo=2, cargo_coords=(0,0,0.5))
    v5 = Voxel(coords=(1,0,1), cargo=2, cargo_coords=(0,0,0))
    v6 = Voxel(coords=(0,1,1), cargo=2, cargo_coords=(0,0,0))
    v7 = Voxel(coords=(1,1,1), cargo=2, cargo_coords=(0,0,0.5))

    voxels = [v0, v1, v2, v3, v4, v5, v6, v7]

    # test creating lattice from unit cell and not from unit cell ✅
    lattice = Lattice(voxels, is_unit_cell=False)
        
    surr = Surroundings(lattice)
    v0_surr = surr.voxel_surroundings(0)

    print(v0_surr)