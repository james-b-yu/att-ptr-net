from typing import Any
from pydantic import BaseModel, Field, root_validator
import xml.etree.ElementTree as ET

from const import CONSTS
from util import get_int_after_underscore, get_str_after_underscore, is_pairwise_disjoint

class Terminal(BaseModel):
    word: str = Field()
    lemma: str = Field()
    pos: str = Field()
    morph: str = Field()
    case: str = Field()
    number: str = Field()
    gender: str = Field()
    person: str = Field()
    degree: str = Field()
    tense: str = Field()
    mood: str = Field()
    
    idx: int = Field()


    def get_sym(self):
        """Return POS and other information about this terminal as a string

        Returns:
            str: POS and other information about this terminal as a string
        """
        return self.pos



class Constituent(BaseModel):
    id: int   = Field()    # node's id within the sentence
    head: int | None = Field()    # node's head word TODO: remove None!!
    yld: set[int] = Field()# node's yield
    sym: str     = Field() # node's symbol

    edge_label: str | None = Field(default=None) # the label for the edge from this constituent to its parent
    parent: int | None = Field(default=None) # the id of the parent node

    is_pre_terminal: bool = Field(default=False)
    children: list[int] = Field(defualt=[])

    @root_validator()
    @classmethod
    def _check_children_types(cls, field_values: dict[str, Any]):
        # if "children" not in field_values.keys():
        #     return field_values

        # for child in field_values["children"]:
        #     if field_values["is_pre_terminal"]:
        #         assert isinstance(child, Terminal), "Expected all children to be of type Terminal"
        #     else:
        #         assert isinstance(child, cls), "Expected all children to be of type Constituent"
        return field_values
    
    @property
    def is_unary(self):
        return len(self.children) == 1 and not self.is_pre_terminal
    
    @property
    def is_empty(self):
        # empty constituents are defined as having the following properties. they are used to make "phantom" vp nodes when the verb is missing
        return self.id < 0 and self.sym == "<EMPTY>" and self.head is None and self.is_pre_terminal and len(self.yld) == 0 and not self.children

class ConstituentTree(BaseModel):
    terminals: dict[int, Terminal]
    constituents: dict[int, Constituent]
    root: int
    is_discontinuous: bool
    has_empty_verb_constituents: bool
    has_unary: bool

    @classmethod
    def _integize(cls):
        num_integers = 1048576
        while True:
            num_integers += 1
            yield num_integers

    @classmethod
    def _create_next_empty_constituent(cls, constituents: dict[int, Constituent]):
        """
        Creates an empty constituent and adds this to the sentence's constituent dict, yielding its id. Empty constituent ids are negative integers.
        """
        num_empty_constituents = 0 # counter to keep empty constituent ids unique
        while True:
            num_empty_constituents -= 1
            assert num_empty_constituents not in constituents, f"Empty constituent id '{num_empty_constituents}' already exists"

            constituents.update({
                num_empty_constituents: Constituent(
                    id=num_empty_constituents,
                    sym="<EMPTY>",
                    head=None,
                    is_pre_terminal=True,
                    yld=set(),
                    children=[]
                )
            })

            yield num_empty_constituents

    @classmethod
    def _find_head_candidates(cls, children: list[int], edge_label: str, pos_list: list[str], constituents: dict[int, Constituent], terminals: dict[int, Terminal]):
        children_matching_edge = [c for c in children if constituents[c].edge_label == edge_label]
        assert set([v for c in children_matching_edge for v in constituents[c].yld]).issubset(set(terminals.keys())), "Head candidates must be terminals"

        if len(children_matching_edge) > 1:
            print(f"Warning: {len(children_matching_edge)} candidate edges exist")

        if not pos_list: # if pos_list is empty, then we don't care about the POS of the head
            head_candidates: list[int] = []
            for c in map(lambda v : constituents[v].head, children_matching_edge):
                assert c is not None, "Head candidate of child must have already been assigned a head"
                head_candidates.append(c)

            yield head_candidates
        
        for pos in pos_list:
            head_candidates: list[int] = []
            for c in map(lambda v : constituents[v].head, children_matching_edge):
                assert c is not None, "Head candidate of child must have already been assigned a head"
                if constituents[c].sym == pos:
                    head_candidates.append(c)
            
            yield head_candidates

    @classmethod
    def _find_head(cls, idx: int, constituents: dict[int, Constituent], terminals: dict[int, Terminal]) -> int | None:
        assert idx in constituents, f"Constituent '{idx}' does not exist"
        assert constituents[idx].sym in CONSTS["head_rules"], f"Constituent symbol '{constituents[idx].sym}' has no head rules"

        for rule in CONSTS["head_rules"].values():
            for dir, label, pos_list in rule:
                for head_candidates in cls._find_head_candidates(constituents[idx].children, label, pos_list, constituents, terminals):
                    if dir == 's' and head_candidates:
                        assert len(head_candidates) == 1, "'s' direction requires unique head candidate"
                        return head_candidates[0]
                    elif dir == 'l' and head_candidates:
                        return min(head_candidates)
                    elif dir == 'r' and head_candidates:
                        return max(head_candidates)
                    
        return None

    @classmethod
    def from_tiger_xml(cls, tree_element: ET.Element):
        # find elements in xml tree
        tree_graph = tree_element.find("graph")
        tree_graph_root_name = get_str_after_underscore(tree_graph.attrib["root"])
        tree_terminals = tree_graph.find("terminals")
        tree_nonterminals = tree_graph.find("nonterminals")

        is_discontinuous = ("discontinuous" in tree_graph.attrib) and (tree_graph.attrib["discontinuous"] == "true") # PROP

        terminals: dict[int, Terminal] = {} # use dict for 1-based indexing # PROP
        constituents: dict[int, Constituent] = {} # PROP

        # parse terminals
        for idx, t in enumerate(tree_terminals.iter("t")):
            assert idx + 1 == get_int_after_underscore(t.attrib["id"]), f"Terminal index '{idx + 1}' does not match index implied by its id '{t.attrib['id']}'"
            term = Terminal(**t.attrib, idx=idx + 1)
            terminals.update({
                idx + 1: term
            })

        # create pre-terminals from terminals: the pre-terminals are constituents with id equal to the terminal's index within the sentence
        for t in terminals.values():
            constituents[t.idx] = Constituent(
                    id=t.idx,
                    head=t.idx,
                    yld={t.idx},
                    sym=t.get_sym(),
                    is_pre_terminal=True,
                    children=[t.idx]
                )
        
        integize_generator = cls._integize()
        integize_dict: dict[str, int] = {}

        # create non-terminal constituents
        for nt in tree_nonterminals.iter("nt"):
            nt_id = get_int_after_underscore(nt.attrib["id"])
            if nt_id is None:
                nt_id = next(integize_generator)
                assert nt_id not in integize_dict.values()
                integize_dict[get_str_after_underscore(nt.attrib["id"])] = nt_id
            assert nt_id is not None

            nt_sym = nt.attrib["cat"]

            nt_children: list[tuple[str, int]] = [] # list of tuple[edge_label, child_id]

            nt_edges = nt.iter("edge")
            for edge in nt_edges:
                edge_id_ref = get_int_after_underscore(edge.attrib["idref"], integize_dict.get(get_str_after_underscore(edge.attrib["idref"]), None))
                edge_label = edge.attrib["label"]
                assert edge_id_ref is not None
                assert edge_id_ref in constituents

                nt_children.append((edge_label, edge_id_ref))

                # add edge label and parent id to child constituent
                assert constituents[edge_id_ref].parent is None, f"Constituent '{edge_id_ref}' already has a parent"
                assert constituents[edge_id_ref].edge_label is None, f"Constituent '{edge_id_ref}' already has an edge label"
                constituents[edge_id_ref].parent = nt_id
                constituents[edge_id_ref].edge_label = edge_label

            nt_children_ylds = [constituents[c_id].yld for _, c_id in nt_children]
            assert is_pairwise_disjoint(*nt_children_ylds), "Yields of children nodes must be pairwise disjoint"

            constituents[nt_id] = Constituent(
                    id=nt_id,
                    sym=nt_sym,
                    head=None,
                    is_pre_terminal=False,
                    yld=set.union(*nt_children_ylds),
                    children=[c_id for _, c_id in nt_children]
                )
        
        # find root
        root = None # PROP
        try:
            root = int(tree_graph_root_name) 
        except Exception:
            root = integize_dict[tree_graph_root_name]
    
        assert constituents[root].yld == set(range(1, len(terminals) + 1)), f"Root constituent must yield entire sentence"

        # attempt to find heads for all phrases that do not have a head already defined by the markup
        has_empty_verb_constituents = False # PROP
        empty_constituent_generator = cls._create_next_empty_constituent(constituents)
        constituents_without_heads = [c for c in constituents.values() if c.head is None]
        for c in constituents_without_heads:
            if c.sym != CONSTS["vroot_symbol"]:
                c.head = cls._find_head(c.id, constituents, terminals)

                # reattach VP or S
                if c.head is None and c.sym in CONSTS["verb_phrase_reattach_symbols"]:
                    c.head = next(empty_constituent_generator)
                    assert constituents[c.head].is_empty
                    has_empty_verb_constituents = True
            else: # vroots have special treatment: copy the head from the unique non-terminal child
                head_candidates = [constituents[cld].head for cld in c.children if not constituents[cld].is_pre_terminal]
                assert len(head_candidates) == 1, "vroot must have exactly one non-terminal child"
                c.head = head_candidates[0]

        # check to see if unary or not
        has_unary = False # PROP
        for c in constituents.values():
            if c.is_unary:
                has_unary = True

        res = cls(
            terminals=terminals,
            constituents=constituents,
            root=root,
            is_discontinuous=is_discontinuous,
            has_empty_verb_constituents=has_empty_verb_constituents,
            has_unary=has_unary
        )
        res._check_constituent_rules()

        return res

    def _check_constituent_rules(self):
        for i in range(1, self.get_num_words() + 1):
            assert i in self.terminals, f"Word {i} not found in terminals dict"
            assert i in self.constituents, f"Word in position {i} should be in constituents dict"

        for id in self.terminals:
            assert (self.terminals[id].idx == id), f"Expected id in dict '{id}' to match id of terminal '{self.terminals[id].idx}'"
        
        for id in self.constituents:
            assert (self.constituents[id].id == id), f"Expected id in dict '{id}' to match id of constituent '{self.constituents[id].id}'"

            if self.constituents[id].is_pre_terminal:
                assert len(self.constituents[id].children) == 1, f"Pre-terminal constituent '{id}' must have exactly one child"
                assert self.constituents[id].children[0] == id, f"Pre-terminal constituent '{id}' must have itself as a child"

        # check to see if constituent rules are satisfied
        for c in self.constituents.values():
            if not c.is_empty:
                assert is_pairwise_disjoint(*map(lambda cld: self.constituents[cld].yld, c.children)), f"Children of constituent '{c.id}' must be pairwise disjoint"
                assert c.yld == set.union(*map(lambda cld: self.constituents[cld].yld, c.children)), f"Yield of constituent '{c.id}' must be equal to union of children yields"

                assert c.head is not None, f"Constituent {c} must have a head"

                if not self.constituents[c.head].is_empty:
                    children_with_same_head = [cld for cld in c.children if self.constituents[cld].head == c.head]
                    assert len(children_with_same_head) == 1, f"Exactly one child of constituent '{c.id}' must have the same head as the constituent. Instead, found children {children_with_same_head}"

    def get_post_order_traversal(self, root_node: int | None = None):
        if root_node is None:
            root_node = self.root
        assert root_node in self.constituents
        if self.constituents[root_node].is_pre_terminal:
            yield root_node
        else:
            for child in self.constituents[root_node].children:
                yield from self.get_post_order_traversal(child)
            yield root_node

    def get_num_words(self):
        return len(self.terminals)



class Dependency(BaseModel):
    head: int # child
    modifier: int # parent
    sym: str = Field(..., regex=r"^[A-Z]+$")


class DependencyTree(BaseModel):
    modifiers: dict[int, dict[int, list[Dependency]]]
    terminals: dict[int, Terminal]
    num_words: int

    @classmethod
    def from_c_tree(cls, c_tree: ConstituentTree):
        num_words = c_tree.get_num_words() # PROP
        modifiers: dict[int, dict[int, list[Dependency]]] = {} # PROP

        attachment_order_map: dict[int, int] = {h: 1 for h in range(1, num_words + 1)}

        for v in c_tree.get_post_order_traversal():
            z = c_tree.constituents[v].sym
            h = c_tree.constituents[v].head
            assert h is not None, "Constituent must have a head"

            j = attachment_order_map[h]

            for u in c_tree.constituents[v].children:
                m = c_tree.constituents[u].head
                assert m is not None
                if m != h:
                    if h not in modifiers:
                        modifiers[h] = {}
                    if j not in modifiers[h]:
                        modifiers[h][j] = []
                    modifiers[h][j].append(Dependency(head=h, modifier=m, sym=z))
            
            attachment_order_map[h] += 1

        return cls(
            modifiers=modifiers,
            terminals=c_tree.terminals,
            num_words=num_words
        )       
    
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

@classmethod
def d_to_c(cls: type[ConstituentTree], d: DependencyTree):
    """
    Creates a constituent tree without unaries and without empty verbs
    """

    used_ids: set[int] = set()
    definitely_not_root: set[int] = set() # set of ids of constituents which have been added as children
    is_discontinuous = False

    constituents: dict[int, Constituent] = {} # id |-> constituent
    phi: dict[int, int] = {} # h |-> id of v

    def counter(start: int):
        the_id = start
        while True:
            the_id += 1
            yield the_id

    non_pre_terminal_id_generator = counter(1000000)

    for h in d.get_post_order_traversal():
        # create pre-terminals
        v_id = h
        assert v_id not in used_ids, "Ran out of ids to use!"
        used_ids.add(v_id)
        definitely_not_root.add(v_id)

        v = Constituent(
            id=v_id,
            head=h,
            yld=set([h]),
            sym=d.terminals[h].get_sym(),
            is_pre_terminal=True,
            children=[h]
            )
        constituents[v_id] = v

        # set phi[h] to be the corresponding pre-terminal
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
            new_id = next(non_pre_terminal_id_generator)
            assert new_id not in used_ids
            used_ids.add(new_id)
            definitely_not_root.update(new_children)
            new_yld = set.union(*[constituents[c].yld for c in new_children])
            new_v = Constituent(
                id=new_id,
                head=h,
                yld=new_yld,
                sym=new_Z,
                children=new_children,
                is_pre_terminal=False
            )
            constituents[new_id] = new_v
            phi[h] = new_id

            # set parent of children to be new constituent
            for c in new_children:
                constituents[c].parent = new_id

            # detect discontinuities
            new_yld_min = min(new_yld)
            new_yld_max = max(new_yld)
            if not is_discontinuous:
                for i in range(new_yld_min, new_yld_max):
                    if (i != new_yld_min) and (i != new_yld_max) and (i not in new_yld):
                        is_discontinuous = True
                        break

    root_candidates = constituents.keys() - definitely_not_root
    assert len(root_candidates) == 1, f"Found multiple possible roots"


    root = root_candidates.pop()
    has_empty_verb_constituents = False # TODO: add better handling
    res = cls(
        terminals=d.terminals,
        constituents=constituents,
        root=root,
        is_discontinuous=is_discontinuous,
        has_empty_verb_constituents=has_empty_verb_constituents,
        has_unary=False
    )
    res._check_constituent_rules()

    return res

ConstituentTree.from_d_tree = d_to_c