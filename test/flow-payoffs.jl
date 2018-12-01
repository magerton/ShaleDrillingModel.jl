@testset "Flow gradients" begin

    pa = dcdp_primitives(:exproy,                β, wp, zspace, Πp, ψspace)
    p0 = dcdp_primitives(:exproy_Dgt0,           β, wp, zspace, Πp, ψspace)
    p1 = dcdp_primitives(:exproy_extend,         β, wp, zspace, Πp, ψspace)
    p2 = dcdp_primitives(:exproy_extend_constr,  β, wp, zspace, Πp, ψspace)
    p3 = dcdp_primitives(:exp,                   β, wp, zspace, Πp, ψspace)
    p7 = dcdp_primitives(:constr,                β, wp, zspace, Πp, ψspace)
    p5 = dcdp_primitives(:constr_onecost,        β, wp, zspace, Πp, ψspace)
    p6 = dcdp_primitives(:extend_constr_onecost, β, wp, zspace, Πp, ψspace)

    p33 = dcdp_primitives(:restricted_q, β, wp, zspace, Πp, ψspace)
    p44 = dcdp_primitives(:restricted_q_Dgt0, β, wp, zspace, Πp, ψspace)

    σ = 0.5
    v0 = [3.66508, -14.91197, 1.83802, 2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302,]
    v1 = [3.66508, -14.91197, 1.83802, 2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302, -0.5,]
    v2 = [         -14.91197,          2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302, -0.5,]
    v7 = [         -14.91197,          2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302,]
    v5 = [         -14.91197,          2.74480, 2.35507, -5.57268, -0.45302,]
    v6 = [         -14.91197,          2.74480, 2.35507, -5.57268, -0.15302, -1.0]

    v33 = [-3.0, -2.0, -0.3, ]
    v44 = [-3.0, -2.0, -2.0, -0.3, ]

    roy = 0.2
    geoid = 2

    @test check_flowgrad(v44, σ, p44, geoid, roy)
    @test check_flowgrad(v33, σ, p33, geoid, roy)
    @test check_flowgrad(v6, σ, p6, geoid, roy)
    @test check_flowgrad(v5, σ, p5, geoid, roy)
    @test check_flowgrad(v7, σ, p7, geoid, roy)

    @test check_flowgrad(v0, σ, pa, geoid, roy)
    @test check_flowgrad(v0, σ, p0, geoid, roy)
    @test check_flowgrad(v1, σ, p1, geoid, roy)
    @test check_flowgrad(v2, σ, p2, geoid, roy)
    @test check_flowgrad(v0[2:end], σ, p3, geoid, roy)
end
