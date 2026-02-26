backbone=dict( 
    type='resnet18-skip01-dec4',
    skip_layers = [0,1],
    dec_rate = 4,
    concat_last_feat = True, 
    skip_connection =True,
    )

nf=dict(
    type='GlowCoupling',
    coupling_blocks=[3, 3],
    clamp_alpha=1.2,
    gamma=0.0,
    kernel_size=3,
    relu_slope=0,
    ch_rate=2.0,
    linear_func=True
    )

