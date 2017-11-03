let p1 = dcdp_primitives(:addlin,     β, wp, zspace, Πp1, ψspace),
    p2 = dcdp_primitives(:add,        β, wp, zspace, Πp1, ψspace),
    p3 = dcdp_primitives(:adddisc,    β, wp, zspace, Πp1, ψspace),
    p4 = dcdp_primitives(:addlincost, β, wp, zspace, Πp1, ψspace),
    p5 = dcdp_primitives(:linbreak,   β, wp, zspace, Πp1, ψspace),
    p6 = dcdp_primitives(:bigbreak,   β, wp, zspace, Πp1, ψspace),
    p7 = dcdp_primitives(:addexp,     β, wp, zspace, Πp1, ψspace),
    p8 = dcdp_primitives(:breakexp,   β, wp, zspace, Πp1, ψspace),
    p9 = dcdp_primitives(:allexp  ,   β, wp, zspace, Πp1, ψspace),
    σ = 0.5 # 14.9407

    @test check_flowgrad([-1.57237, 0.0, 0.5, 0.1, -3.16599, -1.2374, 2.23388], σ, p9, 0.2, 1)
    @test check_flowgrad([-1.57237, 0.0202203, 1.00763, -2.94104, 2.0,    -2.1967, -2.94104, -0.740041, -2.94104, -0.740041, 2.0,     2.34751, 9.25933], σ, p8, 0.2, 1)
    @test check_flowgrad([0.0, 0.0, 0.0, -3.16599, -1.2374, 2.23388], σ, p7, 0.2, 1)
    @test check_flowgrad([-1.19016, 0.0232, 0.91084, -3.16599, -1.2374, 2.23388], σ, p1, 0.2, 1)
    @test check_flowgrad([-1.19016, -3.16599, -1.2374, 2.23388],                  σ, p2, 0.2, 1)
    @test check_flowgrad([-1.19016, 0.91084, -3.16599, -1.2374, 2.23388],         σ, p3, 0.2, 1)
    @test check_flowgrad([-1.19016, 0.91084, -3.16599, -1.2374, 2.23388, 4.0, 1.0], σ, p4, 0.2, 1)
    @test check_flowgrad([-1.19016, 0.91084, -3.16599, -1.2374, 2.23388, 4.0, 1.0], σ, p5, 0.2, 1)
    @test check_flowgrad([-1.57237, 0.0202203, 1.00763, -2.94104,     -2.1967, -2.94104, -0.740041, -2.94104, -0.740041,      2.34751, 9.25933], σ, p6, 0.2, 1)
end
