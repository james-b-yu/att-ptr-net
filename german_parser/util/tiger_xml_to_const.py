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




class Constituent(BaseModel):
    id: int   = Field()    # node's id within the sentence
    head: int | None = Field()    # node's head word TODO: remove None!!
    yld: set[int] = Field()# node's yield
    sym: str     = Field() # node's symbol

    edge_label: str | None = Field(default=None) # the label for the edge from this constituent to its parent
    parent: int | None = Field(default=None) # the id of the parent node

    is_pre_terminal: bool = False
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

class ConstituentTree:
    def _integize(self):
        num_integers = 1048576
        while True:
            num_integers += 1
            yield num_integers

    def _create_next_empty_constituent(self):
        """
        Creates an empty constituent and adds this to the sentence's constituent dict, yielding its id. Empty constituent ids are negative integers.
        """
        num_empty_constituents = 0 # counter to keep empty constituent ids unique
        while True:
            num_empty_constituents -= 1
            assert num_empty_constituents not in self.constituents, f"Empty constituent id '{num_empty_constituents}' already exists"

            self.constituents.update({
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

    def _find_head_candidates(self, children: list[int], edge_label: str, pos_list: list[str]):
        children_matching_edge = [c for c in children if self.constituents[c].edge_label == edge_label]
        assert set([v for c in children_matching_edge for v in self.constituents[c].yld]).issubset(set(self.terminals.keys())), "Head candidates must be terminals"

        if len(children_matching_edge) > 1:
            print(f"Warning: {len(children_matching_edge)} candidate edges exist")

        if not pos_list: # if pos_list is empty, then we don't care about the POS of the head
            yield [v for c in children_matching_edge for v in self.constituents[c].yld]
        
        for pos in pos_list:
            yield [v for c in children_matching_edge for v in self.constituents[c].yld if self.constituents[v].sym == pos]

    def _find_head(self, idx: int) -> int | None:
        assert idx in self.constituents, f"Constituent '{idx}' does not exist"
        assert self.constituents[idx].sym in CONSTS["head_rules"], f"Constituent symbol '{self.constituents[idx].sym}' has no head rules"

        for rule in CONSTS["head_rules"].values():
            for dir, label, pos_list in rule:
                for head_candidates in self._find_head_candidates(self.constituents[idx].children, label, pos_list):
                    if dir == 's' and head_candidates:
                        assert len(head_candidates) == 1, "'s' direction requires unique head candidate"
                        return head_candidates[0]
                    elif dir == 'l' and head_candidates:
                        return min(head_candidates)
                    elif dir == 'r' and head_candidates:
                        return max(head_candidates)
                    
        return None

    def __init__(self, tree_element: ET.Element):
        # find elements in xml tree
        self.tree_graph = tree_element.find("graph")
        self.tree_graph_root_name = get_str_after_underscore(self.tree_graph.attrib["root"])
        self.tree_terminals = self.tree_graph.find("terminals")
        self.tree_nonterminals = self.tree_graph.find("nonterminals")

        self.is_discontinuous = ("discontinuous" in self.tree_graph.attrib) and (self.tree_graph.attrib["discontinuous"] == "true")

        self.terminals: dict[int, Terminal] = {} # use dict for 1-based indexing
        self.constituents: dict[int, Constituent] = {}

        # parse terminals
        for idx, t in enumerate(self.tree_terminals.iter("t")):
            assert idx + 1 == get_int_after_underscore(t.attrib["id"]), f"Terminal index '{idx + 1}' does not match index implied by its id '{t.attrib['id']}'"
            term = Terminal(**t.attrib, idx=idx + 1)
            self.terminals.update({
                idx + 1: term
            })

        # create pre-terminals from terminals: the pre-terminals are constituents with id equal to the terminal's index within the sentence
        self.constituents.update({
                t.idx: Constituent(
                    id=t.idx,
                    head=t.idx,
                    yld={t.idx},
                    sym=t.pos,
                    is_pre_terminal=True,
                    children=[t.idx]
                )
            for t in self.terminals.values()})
        
        self.integize_generator = self._integize()
        integize_dict: dict[str, int] = {}

        # create non-terminal constituents
        for nt in self.tree_nonterminals.iter("nt"):
            nt_id = get_int_after_underscore(nt.attrib["id"])
            if nt_id is None:
                nt_id = next(self.integize_generator)
                assert nt_id not in integize_dict.values()
                integize_dict[get_str_after_underscore(nt.attrib["id"])] = nt_id
            assert nt_id is not None

            nt_sym = nt.attrib["cat"]
            nt_head = None # default to None

            nt_children: list[tuple[str, int]] = [] # list of tuple[edge_label, child_id]

            nt_edges = nt.iter("edge")
            for edge in nt_edges:
                edge_id_ref = get_int_after_underscore(edge.attrib["idref"], integize_dict.get(get_str_after_underscore(edge.attrib["idref"]), None))
                edge_label = edge.attrib["label"]
                assert edge_id_ref is not None
                assert edge_id_ref in self.constituents

                nt_children.append((edge_label, edge_id_ref))
                if edge.attrib["label"] == "HD":
                    nt_head = edge_id_ref

                # add edge label and parent id to child constituent
                assert self.constituents[edge_id_ref].parent is None, f"Constituent '{edge_id_ref}' already has a parent"
                assert self.constituents[edge_id_ref].edge_label is None, f"Constituent '{edge_id_ref}' already has an edge label"
                self.constituents[edge_id_ref].parent = nt_id
                self.constituents[edge_id_ref].edge_label = edge_label

            nt_children_ylds = [self.constituents[c_id].yld for _, c_id in nt_children]
            assert is_pairwise_disjoint(*nt_children_ylds), "Yields of children nodes must be pairwise disjoint"

            self.constituents.update({
                nt_id: Constituent(
                    id=nt_id,
                    sym=nt_sym,
                    head=nt_head,
                    is_pre_terminal=False,
                    yld=set.union(*nt_children_ylds),
                    children=[c_id for _, c_id in nt_children]
                )
            })
        
        # find root
        try:
            self.root = int(self.tree_graph_root_name)
        except Exception:
            self.root = integize_dict[self.tree_graph_root_name]
    
        assert self.constituents[self.root].yld == set(range(1, len(self.terminals) + 1)), f"Root constituent must yield entire sentence"

        # attempt to find heads for all phrases that do not have a head already defined by the markup
        empty_constituent_generator = self._create_next_empty_constituent()
        constituents_without_heads = [c for c in self.constituents.values() if c.head is None]
        for c in constituents_without_heads:
            if c.sym != CONSTS["vroot_symbol"]:
                c.head = self._find_head(c.id)

                # reattach VP or S
                if c.head is None and c.sym in CONSTS["verb_phrase_reattach_symbols"]:
                    c.head = next(empty_constituent_generator)
                    assert self.constituents[c.head].is_empty

                print(f"Found head word {c.head} for constituent {c.id}")
            else: # vroots have special treatment: copy the head from the first S; if no S, then copy from the first VP
                constituent_candidates = [*[cld for cld in c.children if self.constituents[cld].sym == CONSTS["s_symbol"]], *[cld for cld in c.children if self.constituents[cld].sym == CONSTS["vp_symbol"]]]
                if constituent_candidates:
                    c.head =self.constituents[constituent_candidates[0]].head

        # check to see if unary or not
        self.has_unary = False
        for c in self.constituents.values():
            if c.is_unary:
                self.has_unary = True

    
    def check_constituent_rules(self):
        # check to see if constituent rules are satisfied
        for c in self.constituents.values():
            if not c.is_empty:
                assert is_pairwise_disjoint(*map(lambda cld: self.constituents[cld].yld, c.children)), f"Children of constituent '{c.id}' must be pairwise disjoint"
                assert c.yld == set.union(*map(lambda cld: self.constituents[cld].yld, c.children)), f"Yield of constituent '{c.id}' must be equal to union of children yields"

                children_with_same_head = [cld for cld in c.children if self.constituents[cld].head == c.head]
                assert len(children_with_same_head) == 1, f"Exactly one child of constituent '{c.id}' must have the same head as the constituent. Instead, found children {children_with_same_head}"
