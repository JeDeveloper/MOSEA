
import time

from algorithm.lattice.Lattice import Lattice
from algorithm.lattice.Voxel import Voxel, Bond
from algorithm.symmetry.Surroundings import Surroundings
from algorithm.symmetry.SymmetryDf import SymmetryDf

from algorithm.painting.Mesovoxel import Mesovoxel
from algorithm.painting.Painter import Painter

class Moses:
    """the class for painting via the MOSES algorithm"""
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    def run(self):
        """computes both phases of MOSES algorithm, then maps the rest of the lattice"""
        print("Starting MOSES...")
        start_time = time.time()

        # --- INITIALIZE DATA STRUCTURES ---
        # computes all symmetries, filling symmetry_df
        # with all possible voxel pairs and their symmetries
        self.surroundings = Surroundings(self.lattice)
        self.symmetry_df = SymmetryDf(self.lattice, self.surroundings)  # => a useful function
        self.has_symmetry = lambda v1, v2: self.symmetry_df.has_symmetry(v1, v2)

        # initialize the structural voxels
        self.mesovoxel = Mesovoxel(self.lattice, self.has_symmetry)
        self.painter = Painter(self.lattice, self.symmetry_df)
        self.n_colors = 0
        self.uncolored_bonds = self.get_uncolored_bonds()
        self.seen_bonds = set(self.uncolored_bonds)

        self.str_paint()
        self.comp_paint()
        self.reconcile_split_types()
        # self.map_lattice()
        print(f"Done! Took {time.time() - start_time:.2f} seconds.")

    def str_paint(self):
        """paint an initial path of bonds connecting all structural voxels"""
        for voxel1 in self.mesovoxel.structural_voxels:
            voxel1 = self.lattice.get_voxel(voxel1)

            # --- paint path of structural bonds ---
            for vertex, bond1 in voxel1.bonds.items():
                voxel2, bond2 = voxel1.get_partner(vertex)
                # ensure (1) neither bond is colored yet
                # and (2) the other voxel is in the mesovoxel + structural
                if (bond1.color or bond2.color) or (voxel2.id not in self.mesovoxel.structural_voxels):
                    continue
                # paint the new bond
                print(f"\n--- PAINT S_BOND ({self.n_colors+1}) --- \nvoxel_{voxel1.id} ({bond1.get_label()}) <---> voxel_{voxel2.id} ({bond2.get_label()})")
                _ = self.paint_new_bond(bond1, bond2, "structural")

        # also paint self symmetries of all structural voxels
        for sv in self.mesovoxel.structural_voxels:
            sv = self.lattice.get_voxel(sv)
            self.painter.self_sym_paint(sv)

    def comp_paint(self):
        """
        paint all the complementary bonds, slowly adding in complementary 
        voxels (new sub-equivalence class) as needed.
        """
        i = 0
        while i < len(self.uncolored_bonds):

            # get bond / voxel iteration variables
            bond1 = self.uncolored_bonds[i]
            # voxel1 = self.lattice.get_voxel(bond1.voxel)

            i += 1 # early increment for continue safety
            if bond1.color is not None: 
                continue # skip bonds which are already painted

            # get the partner
            voxel2, bond2 = bond1.partner.voxel, bond1.partner

            # CASE 1: VOXEL IS ALREADY MAPPED
            if voxel2.id2:
                pv = self.mesovoxel.get_pv(voxel2.id2) # get the proto-voxel representing the equivalence class
                self.painter.map_paint(pv, voxel2, flip=False)

                # --- paint the new bond if still necessary ---
                if self.paint_new_bond(bond1, bond2, "complementary"):
                    print(f"\n--- PAINT C_BOND ({self.n_colors}) --- \nvoxel_{bond1.voxel.id} ({bond1.get_label()}) <---> voxel_{voxel2.id} ({bond2.get_label()})")
                    self.painter.map_paint(voxel2, pv, flip=False) # map back onto proto_voxel
                continue
            
            # CASE 2: VOXEL NOT MAPPED YET
            sv, cv = self.mesovoxel.get_mesoparents(voxel2) # sv always exists, cv may not
            pv, flip = None, False

            # map either flipping complementary bonds or not based on equivalence class
            # if touching sv, mesovoxel.add_comp_voxel(v2, sv)
            if voxel2.is_touching(sv.id2, type=2): 
                # if we need to add this to the mesovoxel
                if not self.mesovoxel.in_mesovoxel(-sv.id2, type=2):
                    self.mesovoxel.add_comp_voxel(voxel2, sv)
                    pv, flip = sv, True
                    self.add_uncolored_bonds(voxel2.bonds.values())

                self.painter.map_paint(sv, voxel2, flip=True)
                pv, flip = (cv, False) if cv else (sv, True)

            else: # if not touching its sv, map(sv -> v2)
                self.painter.map_paint(sv, voxel2, flip=False)
                pv, flip = sv, False
                voxel2.set_id2(sv.id2)

            # --- paint the new bond if still necessary ---
            if self.paint_new_bond(bond1, bond2, "complementary"):
                print(f"\n--- PAINT C_BOND ({self.n_colors}) --- \nvoxel_{bond1.voxel.id} ({bond1.get_label()}) <---> voxel_{voxel2.id} ({bond2.get_label()})")
                self.painter.map_paint(voxel2, sv, flip) # map back onto proto_voxel

    def map_lattice(self):
        """once we have a finalized mesovoxel, map the unique voxels onto the rest of the lattice"""
        for v in self.lattice.voxels:
            # if self.mesovoxel.in_mesovoxel(v):
            #     continue
            # copy-pasting logic from comp_paint.CASE_2
            sv, cv = self.mesovoxel.get_mesoparents(v)
            if sv is None:
                raise RuntimeError(
                    f"get_mesoparents returned sv=None for voxel id={v.id}, "
                    f"id2={getattr(v,'id2',None)}, coords={getattr(v,'coords',None)}"
                )

            # get_mesoparents picks a parent by structural symmetry (surroundings)
            # only. Two voxels can be structurally symmetric yet require different
            # colorings (the Level-3 case) -- reconcile_split_types() registered a
            # distinct representative for each such coloring. Prefer the
            # representative whose painted pattern is actually consistent with
            # whatever v already carries, so v is not forced under a type it is
            # provably not a rotation of.
            better = self._consistent_representative(v, prefer=(sv, cv))
            if better is not None:
                self.painter.map_paint(better, v)
                v.set_id2(better.id2)
            elif cv and v.is_touching(sv.id2, type=2):
                self.painter.map_paint(cv, v)
                v.set_id2(cv.id2)
            else:
                if v.is_touching(sv.id2, type=2):
                    self.painter.map_paint(sv, v, flip=True)
                    v.set_id2(-sv.id2)
                else:
                    self.painter.map_paint(sv, v)
                    v.set_id2(sv.id2)

        # map_lattice paints faces that were still blank after comp_paint, which
        # can expose further Level-3 splits that were not yet decidable. Repair
        # them the same way.
        self.reconcile_split_types()

        for uv in self.lattice.unit_cell_voxels:
            v = self.lattice.get_voxel(self.lattice.voxel_dict3[uv.id])
            self.painter.map_paint(v, uv)
            uv.set_id2(v.id2)

    def _consistent_representative(self, voxel: Voxel, prefer=()):
        """Return the mesovoxel representative whose painted pattern is a
        rotation of `voxel`'s already-painted faces (so mapping it onto `voxel`
        introduces no conflict), or None if `voxel` carries nothing decisive
        yet. Representatives in `prefer` win ties, then lower |id2|."""
        pattern = self._colored_pattern(voxel)
        if not pattern:
            return None

        prefer_ids = {p.id for p in prefer if p is not None}
        candidates = []
        for vid in self.mesovoxel.all_voxels():
            rep = self.lattice.get_voxel(vid)
            if rep.id == voxel.id:
                return rep  # voxel is itself a representative
            if not self._pattern_needs_split(rep, voxel):
                candidates.append(rep)

        if not candidates:
            return None
        candidates.sort(key=lambda r: (r.id not in prefer_ids, abs(r.id2)))
        return candidates[0]


    # --- reconciliation (bug fix) ---
    def _colored_pattern(self, voxel: Voxel) -> dict:
        """{vertex: color} for the colored faces of `voxel` only."""
        return {vtx: b.color for vtx, b in voxel.bonds.items() if b.color is not None}

    def _rotation_equivalent(self, pattern_a: dict, pattern_b: dict) -> bool:
        """True iff some cube rotation maps the colored pattern `pattern_a`
        exactly onto `pattern_b` (same colors on the same faces, nothing left
        over on either side)."""
        if sorted(pattern_a.values()) != sorted(pattern_b.values()):
            return False
        for label in self.painter.rot_dict.all_rotations:
            rotated = self.painter.rot_dict.rotate_bonds(dict(pattern_a), label)
            if {vtx: c for vtx, c in rotated.items()} == pattern_b:
                return True
        return False

    def reconcile_split_types(self):
        """Post-process comp_paint's output to repair mis-assigned voxel types.

        comp_paint's CASE 1 and CASE 2 'else' branches commit a newly seen
        voxel to its structural parent's id2 *before* the parent is fully
        painted, on the assumption that the voxel is a pure rotation of that
        parent. When a complementary bond later forces a color the parent's
        symmetry pattern does not account for, that assumption is silently
        broken: two voxels whose 6-face patterns are provably NOT related by
        any rotation end up sharing one id2 ("type"), and the minted color can
        be left with no complementary color on any representative -- an
        unsatisfiable design (e.g. zinc_blende).

        Only CASE 2's 'if' branch performs the Level-3 split the MOSES paper
        describes (via add_comp_voxel); the other paths do not. Now that
        comp_paint has finished and every representative is fully painted, the
        "is this really a rotation of my type's representative?" test is
        finally reliable, so run it here: every distinct divergent pattern
        gets promoted to its own structural voxel type.
        """
        changed = True
        while changed:
            changed = False
            for v in self.lattice.voxels:
                if not v.id2 or not self.mesovoxel.in_mesovoxel(v.id2, type=2):
                    continue
                if self.mesovoxel.contains_voxel(v):
                    continue  # v is itself a representative
                proto = self.mesovoxel.get_pv(v.id2)
                if proto.id == v.id or not self._pattern_needs_split(proto, v):
                    continue

                # v's painted pattern is provably not a rotation of its assigned
                # type's representative. First see whether it actually matches
                # some *other* existing representative (MOSES just filed it under
                # the wrong one) -- if so, just move it, no new type needed.
                pattern = self._colored_pattern(v)
                moved = False
                for rid in self.mesovoxel.all_voxels():
                    rep = self.lattice.get_voxel(rid)
                    if rep.id == v.id:
                        continue
                    if self._rotation_equivalent(pattern, self._colored_pattern(rep)):
                        v.set_id2(rep.id2)
                        moved = True
                        break
                if moved:
                    changed = True
                    continue

                # genuinely new coloring -> its own structural type (the
                # Level-3 split CASE 2's 'if' branch does, that the other
                # branches skip).
                self.mesovoxel.add_structural_voxel(v)
                self.painter.self_sym_paint(v)
                changed = True

        self._warn_unpaired_representative_colors()

    def _warn_unpaired_representative_colors(self):
        """Sanity check: after reconciliation every color used on a
        representative should have its complement on some representative,
        otherwise the exported design cannot close."""
        used = set()
        for vid in self.mesovoxel.all_voxels():
            for b in self.lattice.get_voxel(vid).bonds.values():
                if b.color:
                    used.add(b.color)
        unpaired = sorted(c for c in used if -c not in used)
        if unpaired:
            print(f"WARNING: representative colors {unpaired} have no complement "
                  f"on any representative -- exported design will not close. "
                  f"reconcile_split_types() could not fully repair this lattice.")

    # --- utils ---
    def _pattern_needs_split(self, parent: Voxel, child: Voxel) -> bool:
        """True iff `child`'s currently-painted 6-face color pattern provably
        cannot be a rotation of `parent`'s: every cube rotation that could align
        them collides on some face they have both already painted.

        If any rotation is collision-free we return False -- `child` may simply
        be an as-yet-incompletely-painted rotation of `parent` (e.g. `parent`
        itself still has uncolored faces), and splitting it off would create a
        redundant voxel type. We only split when it is impossible for them to be
        the same type.
        """
        child_colors = [b.color for b in child.bonds.values()]
        if all(c is None for c in child_colors):
            return False

        for label in self.painter.rot_dict.all_rotations:
            rotated_parent = self.painter.rot_dict.rotate_bonds(parent.bonds, label)
            collision = False
            for vtx, pbond in rotated_parent.items():
                cbond = child.bonds.get(vtx)
                if cbond is None:
                    continue
                if pbond.color is not None and cbond.color is not None \
                        and pbond.color != cbond.color:
                    collision = True
                    break
            if not collision:
                return False # a consistent alignment exists -> not provably new
        return True

    def paint_new_bond(self, bond1: Bond, bond2: Bond, type:str="structural") -> int:
        """paints the new color connecting bond1 and bond2 only if they're not none
        and also paints with self symmetries to exploit this new color
        
        returns 1 if success 0 if not"""
        # --- paint the new bond if still necessary ---
        if bond1.color is not None or bond2.color is not None:
            return 0

        self.n_colors += 1
        self.painter.paint_bonds(bond1, bond2, self.n_colors, type)

        # also paint with self-symmetries
        self.painter.self_sym_paint(bond1.voxel)
        self.painter.self_sym_paint(bond2.voxel)
        return 1
    
    def get_uncolored_bonds(self) -> list[Bond]:
        """get all uncolored bonds in the mesovoxel"""
        voxels = set(self.mesovoxel.all_voxels())
        bonds = set()
        bond_queue = []

        for v in voxels:
            voxel = self.lattice.get_voxel(v)
            for vertex, bond in voxel.bonds.items():
                if bond.color is None:
                    bond_queue.append(bond)
                    bonds.add((bond.voxel.id, vertex))
        
        return bond_queue
    
    def add_uncolored_bonds(self, bonds: list[Bond]):
        for b in bonds:
            if b.color is None and b not in self.seen_bonds:
                self.uncolored_bonds.append(b) 
                self.seen_bonds.add(b)
    
    def unique_voxels(self) -> list[Voxel]:
        """Get all unique voxels in the Mesovoxel"""
        return [self.lattice.get_voxel(v) for v in self.mesovoxel.all_voxels()]