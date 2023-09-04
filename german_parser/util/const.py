from os import path

CONSTS = {
    "model_dir": f"{path.dirname(__file__)}/../../models",
    "data_dir": f"{path.dirname(__file__)}/../../data",
    "none_label": "--", # tiger treebank uses "--" as an empty label
    "head_rules": { 'S': [('s','HD',[])],
                              'VP': [('s','HD',[])],
                              'VZ': [('s','HD',[])],
                              'AVP':[('s','HD',[]),('s','PH',[]),('r','AVC',['ADV']),('l','AVC',['FM'])],
                              'AP': [('s','HD',[]),('s','PH',[])],
                              'DL': [('s','DH',[])],
                              'AA': [('s','HD',[])],
                              'ISU':[('l','UC',[])],
                              'PN': [('r','PNC',['NE','NN','FM','TRUNC','APPR','APPRART','CARD','VVFIN','VAFIN','ADJA','ADJD','XY'])],
                              'MPN': [('r','PNC',['NE','NN','FM','TRUNC','APPR','APPRART','CARD','VVFIN','VAFIN','ADJA','ADJD','XY'])],
                              'NM': [('r','NMC',['NN','CARD'])],
                              'MTA':[('r','ADC',['ADJA'])],
                              'PP': [('r','HD',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','AC',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('r','PH',['APPRART','APPR','APPO','PROAV','NE','APZR','PWAV','TRUNC']),('l','NK',['PROAV'])],
                              'CH': [('s','HD',[]),('l','UC',['FM','NE','XY','CARD','ITJ'])],
                              'NP': [('l','HD',['NN']),('l','NK',['NN']),('r','HD',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT']),('r','NK',['NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT']),('r','PH',['NN','NE','PPER','PIS','PDS','PRELS','PRF','PWS','PPOSS','FM','TRUNC','ADJA','CARD','PIAT','PWAV','PROAV','ADJD','ADV','APPRART','PDAT'])],              
                              'CAC':[('l','CJ',[])],
                              'CAP':[('l','CJ',[])],
                             'CAVP':[('l','CJ',[])],
                              'CCP':[('l','CJ',[])],
                              'CNP':[('l','CJ',[])],
                              'CO': [('l','CJ',[])],
                              'CPP':[('l','CJ',[])],
                              'CS': [('l','CJ',[])],
                              'CVP':[('l','CJ',[])],
                              'CVZ':[('l','CJ',[])]
                            },
        "verb_phrase_reattach_symbols": ["S", "VP"],
        "vroot_symbol": "VROOT",
        "d_tree_root_sym": "DROOT",
        "s_symbol": "S",
        "vp_symbol": "VP",
}

"""
headrules is a dict[str, rules]
where rules is list[rule]
where rule is tuple[direction, edge_label, list[str]]
    edge_label dictates the edge to follow
    list[str] is a list of parts of speech to find along that edge
    direction is one of 'l' or 'r' for left or right: if left, follow the edge to constituent that matches the first pos in the list, if right, follow the edge to the rightmost constituent that matches the pos
"""