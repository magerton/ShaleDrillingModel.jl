@testset "Flow gradients" begin

    p0 = dcdp_primitives(:exproy_Dgt0,          β, wp, zspace, Πp, ψspace)
    p1 = dcdp_primitives(:exproy_extend,        β, wp, zspace, Πp, ψspace)
    p2 = dcdp_primitives(:exproy_extend_constr, β, wp, zspace, Πp, ψspace)

    σ = 1.66561
    v0 = [3.66508, -14.91197, 1.83802, 2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302,]
    v1 = [3.66508, -14.91197, 1.83802, 2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302, -0.5,]
    v2 = [         -14.91197,          2.74480, 2.35507, -6.57268, -4.91350, 2.41477, -0.45302, -0.5,]

    roy = 0.2
    geoid = 2

    @test check_flowgrad(v0, σ, p0, geoid, roy)
    @test check_flowgrad(v1, σ, p1, geoid, roy)
    @test check_flowgrad(v2, σ, p2, geoid, roy)
end
