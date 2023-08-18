from tiger_xml_to_const import ConstituentTree, Constituent
from pydantic import BaseModel, Field, root_validator

class Dependency(BaseModel):
    head: int # child
    modifier: int # parent
    sym: str = Field(..., regex=r"^[A-Z]+$")


class DependencyTree:
    def __init__(self, c_tree: ConstituentTree):
        self.num_words = c_tree.get_num_words()
        self.modifiers: dict[int, dict[int, list[Dependency]]] = {}

        attachment_order_map: dict[int, int] = {h: 1 for h in range(1, self.num_words + 1)}

        for v in c_tree.get_post_order_traversal():
            z = c_tree.constituents[v].sym
            h = c_tree.constituents[v].head
            assert h is not None, "Constituent must have a head"

            j = attachment_order_map[h]

            for u in c_tree.constituents[v].children:
                m = c_tree.constituents[u].head
                assert m is not None
                if m != h:
                    if h not in self.modifiers:
                        self.modifiers[h] = {}
                    if j not in self.modifiers[h]:
                        self.modifiers[h][j] = []
                    self.modifiers[h][j].append(Dependency(head=h, modifier=m, sym=z))
            
            attachment_order_map[h] += 1
    
    def get_arcs(self):
        return [a for v in self.modifiers.values() for m in v.values() for a in m]

    def get_words(self):
        all_words: set[int] = set()

        for a in self.get_arcs():
            all_words.add(a.head)
            all_words.add(a.modifier)
        
        return all_words

    def get_root(self):
        all_words = self.get_words()
        modifier_words: set[int] = set()

        for a in self.get_arcs():
            # modifier words (m) are children; head words (h) are parents
            modifier_words.add(a.modifier)

        root_set = all_words - modifier_words
        assert len(root_set) == 1

        return root_set.pop()
    

    def get_tree_map(self):
        tree_map: dict[int, set[int]] = {}

        for arc in self.get_arcs():
            parent = arc.head
            child = arc.modifier

            if parent not in tree_map:
                tree_map[parent] = set([child])
            else:
                assert child not in tree_map[parent], "Modifier must not already be assigned as child of parent (head)"
                tree_map[parent].add(child)

        return tree_map
    

    def _get_post_order_traversal(self, tree_map: dict[int, set[int]], root: int):
        if root not in tree_map or not tree_map[root]:
            yield root
        else:
            for child in tree_map[root]:
                yield from self._get_post_order_traversal(tree_map, child)
                
            yield root

    def get_post_order_traversal(self):
        yield from self._get_post_order_traversal(self.get_tree_map(), self.get_root())

def d_to_c(d: DependencyTree):
    id_to_constituent: dict[int, Constituent] = {}
    phi: dict[int, int] = {} # h |-> id of v

    res: list[Constituent] = []

    def counter():
        the_id = 0
        while True:
            the_id += 1
            yield the_id

    id_generator = counter()

    for h in d.get_post_order_traversal():
        # create pre-terminals
        v_id = next(id_generator)
        v = Constituent(
            id=v_id,
            head=h,
            yld=set([h]),
            sym="<POS>",
            pre_terminal=True,
            children=[]
            )
        id_to_constituent[v_id] = v
        # set phi[h] to be the corresponding preterminal
        phi[h] = v_id

        # create sorted list of modifiers by priority order
        Mh: list[list[Dependency]] = []
        if h in d.modifiers:
            keys_sorted = sorted(d.modifiers[h].keys())
            Mh = [d.modifiers[h][j] for j in keys_sorted]

        for j in range(len(Mh)):
            new_Z = Mh[j][0].sym # label shared by group of modifiers
            # assert heads h and labels Z are the same in all Mh[j]
            for k in Mh[j]:
                assert k.head == h
                assert k.sym == new_Z


            # create new constituent with yield as the union of phi[h] and phi[m] yields
            new_children = [phi[h], *[phi[m.modifier] for m in Mh[j]]]
            new_id = next(id_generator)
            new_yld = set.union(*[id_to_constituent[c].yld for c in new_children])
            new_v = Constituent(
                id=new_id,
                head=h,
                yld=new_yld,
                sym=new_Z,
                children=new_children
            )
            id_to_constituent[new_id] = new_v

            res.append(new_v)
            phi[h] = new_id
    return res