import spacy
spacy.require_gpu()
import nltk.corpus.europarl_raw as er
from spacy import displacy

import pandas as pd

from dataclasses import dataclass, field

@dataclass
class TagType:
    required_morph_attrs: set[str] = field(default_factory=set) # set of all morph attributes that are required for this tag
    poses: dict[str, float] = field(default_factory=dict) # set of all parts of speech that this tag can take

    morph_attrs: dict[str, float] = field(default_factory=dict) # dict of all morph attributes and their relative frequency
    child_tags: dict[str, float] = field(default_factory=dict) # dict of all child tags and their relative frequency
    child_poses: dict[str, float] = field(default_factory=dict) # dict of all child parts of speech and their relative frequency
    deps: dict[str, float] = field(default_factory=dict) # dict of all dependencies and their relative frequency
    
    num_occurances: int = 0 # number of times this tag occurs in the corpus
    freq: float = 0.0 # relative frequency of this tag in the corpus



def get_morph_types(nlp: spacy.Language, sents: list[str])  -> tuple[dict[str, TagType], dict[str, set[str]]]:
    tag_map: dict[str, TagType] = {} # map of all tags to their TagType
    morph_attr_map: dict[str, set[str]] = {} # map of all morph attributes to the set of all values they take

    num_sents = len(sents)
    print(f"Processing {num_sents} sentences")
    docs = list(nlp.pipe(sents, batch_size=1000, n_process=1))
    print(f"Processed {num_sents} sentences")

    for doc in docs:
        for token in doc:
            tag = token.tag_
            morph = token.morph
            pos = token.pos_

            morph_dict = morph.to_dict()
            morph_attrs = set(morph_dict.keys())

            deps = {v.dep_: 1.0 for v in token.children}
            child_tags = {v.tag_ for v in token.children}
            child_poses = {v.pos_ for v in token.children}

            if tag not in tag_map:
                tag_map[tag] = TagType(morph_attrs={v: 1 for v in morph_attrs}, deps=deps, num_occurances=1, poses={pos: 1.0}, child_tags={v: 1.0 for v in child_tags}, child_poses={v: 1.0 for v in child_poses})
            else:
                tag_map[tag].num_occurances += 1
                
                if pos not in tag_map[tag].poses:
                    tag_map[tag].poses[pos] = 1.0
                else:
                    tag_map[tag].poses[pos] += 1.0

                for attr in morph_attrs:
                    if attr not in tag_map[tag].morph_attrs:
                        tag_map[tag].morph_attrs[attr] = 1.0
                    else:
                        tag_map[tag].morph_attrs[attr] += 1.0

                for dep in deps:
                    if dep not in tag_map[tag].deps:
                        tag_map[tag].deps[dep] = 1.0
                    else:
                        tag_map[tag].deps[dep] += 1.0

                for child_tag in child_tags:
                    if child_tag not in tag_map[tag].child_tags:
                        tag_map[tag].child_tags[child_tag] = 1.0
                    else:
                        tag_map[tag].child_tags[child_tag] += 1.0

                for child_pos in child_poses:
                    if child_pos not in tag_map[tag].child_poses:
                        tag_map[tag].child_poses[child_pos] = 1.0
                    else:
                        tag_map[tag].child_poses[child_pos] += 1.0

            for attr in morph_attrs:
                if attr not in morph_attr_map:
                    morph_attr_map[attr] = set()

                morph_attr_map[attr].add(morph_dict[attr])

    total_tokens = 0
    for tag in tag_map:
        tag_map[tag].required_morph_attrs = {v for v in tag_map[tag].morph_attrs if tag_map[tag].morph_attrs[v] == tag_map[tag].num_occurances}

        for pos in tag_map[tag].poses:
            tag_map[tag].poses[pos] = 0.01 * ((100 * tag_map[tag].poses[pos]) // tag_map[tag].num_occurances)

        for attr in tag_map[tag].morph_attrs:
            tag_map[tag].morph_attrs[attr] = 0.01 * ((100 * tag_map[tag].morph_attrs[attr]) // tag_map[tag].num_occurances)

        for dep in tag_map[tag].deps:
            tag_map[tag].deps[dep] = 0.01 * ((100 * tag_map[tag].deps[dep]) // tag_map[tag].num_occurances)

        for child_tag in tag_map[tag].child_tags:
            tag_map[tag].child_tags[child_tag] = 0.01 * ((100 * tag_map[tag].child_tags[child_tag]) // tag_map[tag].num_occurances)

        for child_pos in tag_map[tag].child_poses:
            tag_map[tag].child_poses[child_pos] = 0.01 * ((100 * tag_map[tag].child_poses[child_pos]) // tag_map[tag].num_occurances)

        total_tokens += tag_map[tag].num_occurances
    
    for tag in tag_map:
        tag_map[tag].freq = 0.01 * ((100 * tag_map[tag].num_occurances) // total_tokens)

    return tag_map, morph_attr_map



if __name__ == '__main__':
    nlp = spacy.load("de_dep_news_trf")
    sents: list[str] = [" ".join(v) for v in er.german.sents()][:] #type: ignore
    tag_map, morph_attr_map = get_morph_types(nlp, sents)
    

    all_tags = tag_map.keys()
    morph_attrs = [tag_map[t].morph_attrs for t in all_tags]
    required_morph_attrs = [tag_map[t].required_morph_attrs for t in all_tags]
    deps = [tag_map[t].deps for t in all_tags]
    poses = [tag_map[t].poses for t in all_tags]
    child_tags = [tag_map[t].child_tags for t in all_tags]
    child_poses = [tag_map[t].child_poses for t in all_tags]
    desc = [spacy.explain(t) for t in all_tags]

    tag_df = pd.DataFrame({
        "tag": all_tags,
        "desc": desc,
        "required_morph_attrs": required_morph_attrs,
        "morph_attrs": morph_attrs,
        "poses": poses,
        "deps": deps,
        "child_tags": child_tags,
        "child_poses": child_poses
    })

    print(tag_df)

    all_morphs = morph_attr_map.keys()
    morph_vals = [morph_attr_map[m] for m in all_morphs]

    morph_df = pd.DataFrame({
        "morph": all_morphs,
        "vals": morph_vals
    })

    print(morph_df)