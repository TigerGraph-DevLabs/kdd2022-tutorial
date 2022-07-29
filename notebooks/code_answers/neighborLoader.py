train_loader = conn.gds.neighborLoader(
    v_in_feats={"Transaction": ["seller_k_size", "buyer_k_size"], 
                "NFT_User": ["pagerank", "kcore_size"], 
                "NFT": ["fastrp_embedding"], 
                "NFT_Collection": ["fastrp_embedding"], 
                "Category": ["fastrp_embedding"]},
    v_out_labels={"Transaction": ["usd_price"]},
    v_extra_feats={"Transaction":  ["train"]},
    filter_by={"Transaction": "train"},
    shuffle=True,
    batch_size=2048,
    buffer_size=4,
    add_self_loop=True,
    reverse_edge=True
)