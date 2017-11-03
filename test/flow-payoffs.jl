
let p1 = dcdp_primitives(:lin,      β, wp, zspace, Πp1, ψspace),
    p2 = dcdp_primitives(:exp,      β, wp, zspace, Πp1, ψspace),
    p3 = dcdp_primitives(:breakexp, β, wp, zspace, Πp1, ψspace),
    p4 = dcdp_primitives(:breaklin, β, wp, zspace, Πp1, ψspace),
    σ = 0.5,
    v1 = [-0x1.2d34868c62f72p+0, 1.0,                  0x1.2b663d526e945p-6,  0x1.e541149a44256p-1,-0x1.a46b48dd5571bp+1,-0x1.55350321e88e3p+0, 0x1.1ec8d5020b7eep+1],
    v2 = [-0x1.2197680e926bap-1,-0x1.92d942f84728p-1,  0x1.bd8611a4e2182p-4,  0x1.bfbaccea9e265p+0,-0x1.d07733f0b209cp+2,-0x1.4c8849c43b4fp+2,  0x1.37c9b94161073p+1],
    v3 = [-0x1.2d34868c62f72p+0, 1.0,                  0x1.2b663d526e945p-6,  0x1.e541149a44256p-1,-0x1.a46b48dd5571bp+1,-0x1.55350321e88e3p+0, 0x1.1ec8d5020b7eep+1, -0x1.2d34868c62f72p+0, 1.0,                  0x1.2b663d526e945p-6,0x1.e541149a44256p-1,-0x1.a46b48dd5571bp+1,-0x1.55350321e88e3p+0],
    v4 = [-0x1.2197680e926bap-1,-0x1.92d942f84728p-1,  0x1.bd8611a4e2182p-4,  0x1.bfbaccea9e265p+0,-0x1.d07733f0b209cp+2,-0x1.4c8849c43b4fp+2,  0x1.37c9b94161073p+1, -0x1.2197680e926bap-1,-0x1.92d942f84728p-1,  0x1.bd8611a4e2182p-4,0x1.bfbaccea9e265p+0,-0x1.d07733f0b209cp+2,-0x1.4c8849c43b4fp+2 ]

    @test check_flowgrad(v1, σ, p1, 0.2, 1)
    @test check_flowgrad(v2, σ, p2, 0.2, 1)
    @test check_flowgrad(v3, σ, p3, 0.2, 1)
    @test check_flowgrad(v4, σ, p4, 0.2, 1)
end
