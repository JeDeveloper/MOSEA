from algorithm.lattice.Voxel import Voxel
from algorithm.lattice.Lattice import Lattice
from typing import Callable, Any
from collections import defaultdict, deque

class Mesovoxel:
    def __init__(self, lattice: Lattice, has_symmetry: Callable[[Any, Any], tuple[bool, list]]):
        """
        Mesovoxel data structure, which is comprised of two sets
        
        (1) structural voxels:      clearly defined by symmetry alone
        (2) complementary voxels:   starts empty, and we add voxels to it
                                    1-by-1 as we paint bonds
        """
        # parent lattice/painter classes
        self.lattice = lattice
        self.has_symmetry = has_symmetry

        # these two sets uniquely define the mesovoxel
        # can be indexed with id2-1
        self.structural_voxels, self.adj_list = self.init_structural_voxels()
        self.complementary_voxels: list[int] = []

    def init_structural_voxels(self) -> tuple[list[int], dict[int, list[int]]]:
        """
        Initialize a list of structural voxels based on the "lattice" attribute.
        Returns:
            structural_voxels: A set of voxel ids (ints) of structural voxels in lattice
            adj_list: the adjacency list mapping {id2: [v.id1, v.id1, ...]} 
                      where adj_list[id2][0] is the proto-voxel
        """
        # iterate over voxels
        voxels = iter(self.lattice.voxels)

        # init with first voxel in lattice
        v_0 = next(voxels)
        v_0.set_id2(1)

        # fill in the data structures with v_0
        structural_voxels = [v_0.id]
        
        i = 2
        for voxel in voxels:
            had_sym = False
            for sv in structural_voxels:
                has_sym, _ = self.has_symmetry(voxel, sv)
                if has_sym: # skip if voxel has symmetry with something in sv
                    # sv = self.lattice.get_voxel(sv)
                    # adj_list[sv.id2].append(voxel.id)
                    had_sym = True
                    break
            if had_sym:
                continue
            # add a new structural voxel to our list
            # voxel.set_id2(i)
            structural_voxels.append(voxel.id)
            i += 1

        # get a CONNECTED SET of structural voxels
        structural_voxels = self.connect_sv2(structural_voxels)

        # initialize the adjacency list
        adj_list = defaultdict(list)
        for i, v_id in enumerate(structural_voxels):
            v = self.lattice.get_voxel(v_id)
            v.set_id2(i+1)
            adj_list[v.id2].append(v.id)

        return structural_voxels, adj_list
    
    def connect_sv2(self, structural_voxels: list[int]) -> list[int]:
        print("Finding a connected set of structural voxels...")

        structural_voxels = list(structural_voxels)
        seed = structural_voxels[0]
        target = set(structural_voxels)

        # part 1: find connected subset containing the seed
        q = deque([seed])
        visited = {seed}
        connected = {seed}

        while q:
            v = q.popleft()
            v = self.lattice.get_voxel(v)
            for bond in v.bonds.values():
                pv = bond.get_partner_voxel()
                if pv is None:
                    continue
                if pv.id in target and pv.id not in visited:
                    visited.add(pv.id)
                    connected.add(pv.id)
                    q.append(pv.id)
        
        unconnected = target - connected

        # part 2: expand outward from the connected frontier
        # swapping connected voxels which are symmetric to unconnected ones
        frontier = deque(list(connected))
        visited = set(connected)

        while unconnected and frontier:
            v = self.lattice.get_voxel(frontier.popleft())

            for bond in v.bonds.values():
                pv = bond.get_partner_voxel()
                if pv is None or pv.id in visited:
                    continue

                visited.add(pv.id)
                frontier.append(pv.id)

                # check if partner voxel can replace anything in unconnected
                for uc in list(unconnected):
                    uc = self.lattice.get_voxel(uc)
                    has_sym, _ = self.has_symmetry(pv, uc)
                    if has_sym:
                        print(f"Replacing unconnected voxel {uc.id} with {pv.id}")
                        unconnected.remove(uc.id)
                        connected.add(pv.id)
                        break
        
        if unconnected:
            raise RuntimeError(
                f"Could not find connected symmetric replacements for {len(unconnected)} "
                f"unconnected structural voxels: {sorted(list(unconnected))[:10]}..."
            )
    
        print("Done!")
        return list(connected)

    
    def connect_sv(self, structural_voxels: list[int]) -> list[int]:
        """Given a set of structural voxels, reshuffle things
        around until we choose a set which is all connected."""
        print("Finding a connected set of structural voxels...")
        structural_voxels = list(structural_voxels)
        v = structural_voxels.pop(0)
        v = self.lattice.get_voxel(v)
        open_bonds = [b for b in v.bonds.values()]
        connected = {v.id}

        while len(open_bonds) > 0:
            bond = open_bonds.pop(0)
            pv = bond.get_partner_voxel()
            if pv.id in structural_voxels:
                open_bonds.extend([b for b in pv.bonds.values() if b not in open_bonds])
                connected.add(pv.id)
        
        unconnected = set(structural_voxels) - connected

        open_bonds = []

        for v in connected:
            v = self.lattice.get_voxel(v)
            open_bonds.extend([b for b in v.bonds.values()])
        
        while len(unconnected) > 0:
            bond = open_bonds.pop(0)
            pv = bond.get_partner_voxel()
            
            # check if partner voxel can replace anything in unconnected
            for uc in unconnected:
                uc = self.lattice.get_voxel(uc)
                has_sym, _ = self.has_symmetry(pv, uc)
                if has_sym:
                    print(f"Replacing unconnected voxel {uc.id} with {pv.id}")
                    unconnected.remove(uc.id)
                    connected.add(pv.id)
                    open_bonds.extend([b for b in pv.bonds.values() if b not in open_bonds])
                    break
        
        print("Done!")
        
        return list(connected)
    

    def in_mesovoxel(self, voxel: Voxel|int, type=1) -> bool:
        """Returns whether the given voxel is in one of two mesovoxel sets or not."""
        if type==1:
            voxel_id = voxel.id if isinstance(voxel, Voxel) else voxel
            return voxel_id in self.structural_voxels or voxel_id in self.complementary_voxels
        elif type==2:
            voxel_id = voxel.id2 if isinstance(voxel, Voxel) else voxel
            in_meso = self.adj_list.get(voxel_id)
            return True if in_meso else False

    def get_mesoparents(self, voxel: Voxel|int) -> list[Voxel, Voxel]:
        """
        Find the best parent voxel in the mesovoxel for the given voxel.
        Voxels satisfying this will either be added to the parent voxel's maplist
        or will become its complementary voxel. Requires no prior id2 information.

        Args:
            voxel (Voxel/int): Voxel to find mesoparent of
        Returns:
            mesoparents: [str_voxel, comp_voxel | None] 
            NOTE: should this return id1 or id2?
        """
        mesoparents = [None, None]

        #TODO: implement the less-naive way to find this
        # find the structural voxel with symmetry to the given voxel
        for s_voxel in self.structural_voxels:
            has_sym, _ = self.has_symmetry(s_voxel, voxel)
            if has_sym:
                mesoparents[0] = self.lattice.get_voxel(s_voxel)
                break

        # also find the complementary voxel
        for c_voxel in self.complementary_voxels:
            has_sym, _ = self.has_symmetry(c_voxel, voxel)
            if has_sym:
                mesoparents[1] = self.lattice.get_voxel(c_voxel)

        return mesoparents
    
    def get_pv(self, id2: int) -> Voxel:
        """given an id2, gets the corresponding proto-voxel"""
        v_id = self.adj_list[id2][0]
        return self.lattice.get_voxel(v_id)
    
    def add_structural_voxel(self, voxel: Voxel) -> int:
        """Register `voxel` as the sole representative of a brand-new structural
        voxel type, returning its new (positive) id2.

        Used when a voxel provably cannot be expressed as a pure rotation of
        its structural parent -- painting one of its complementary bonds forced
        in a color that no symmetry of the parent accounts for -- so forcing it
        under the parent's id2 (as comp_paint's CASE 2 'else' branch used to do
        unconditionally) is wrong: it groups two non-rotation-equivalent voxels
        as one "type" and can leave the minted color with no complement on any
        representative. This is the Level-3 split the MOSES paper describes for
        the case CASE 2's 'if' branch (add_comp_voxel) does not cover.
        """
        new_id2 = max((k for k in self.adj_list if k > 0), default=0) + 1
        print(f"adding structural voxel (id={voxel.id}, id2={new_id2}) -- "
              f"not a rotation of its structural parent")
        voxel.set_id2(new_id2)

        self.adj_list[new_id2] = [voxel.id]
        if voxel.id not in self.structural_voxels:
            self.structural_voxels.append(voxel.id)
        return new_id2

    def add_comp_voxel(self, comp_voxel: Voxel, str_voxel: Voxel):
        """adds the comp_voxel for the specified str_voxel"""
        print(f"adding complementary voxel (id={comp_voxel.id}, id2={-str_voxel.id2})")
        id2 = -str_voxel.id2
        comp_voxel.set_id2(id2)

        # append the new comp_voxel to the data structures
        self.adj_list[id2] = [comp_voxel.id]
        if comp_voxel.id not in self.complementary_voxels:
            self.complementary_voxels.append(comp_voxel.id)

    def contains_voxel(self, voxel: Voxel|int):
        """
        Check if the the given voxel (id/Voxel) is mapped to a voxel in the Mesovoxel
        eg, whether the mesovoxel 'contains' the supplied voxel
        """
        voxel_id = voxel.id if isinstance(voxel, Voxel) else voxel
        return True if voxel_id in self.structural_voxels or voxel_id in self.complementary_voxels else False
    

    def all_voxels(self) -> list[int]:
        """
        Returns the set of all voxels in the mesovoxel. Aka just the current
        structural and complementary voxels.
        """
        return self.structural_voxels + self.complementary_voxels