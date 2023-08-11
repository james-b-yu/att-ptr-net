from tiger_xml_to_const import ConstituentTree
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
        pass