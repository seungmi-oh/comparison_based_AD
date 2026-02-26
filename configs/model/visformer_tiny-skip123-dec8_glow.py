backbone=dict( 
    type='visformer_tiny-skip123-dec8',
    skip_layers = [1,2,3],
    dec_rate = 8,
    concat_last_feat = True, 
    skip_connection =True
    )

nf=dict(
    type='GlowCoupling',
    coupling_blocks=6,
    clamp_alpha=1.9,
    gamma=0.0,
    kernel_size=3,
    relu_slope=0,
    ch_rate=2.0,
    linear_func=True
    )

