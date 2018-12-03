@testset "Flow gradients" begin

    p3   = dcdp_primitives( :one_restr   , β, wp, zspace, Πp, ψspace)
    p4a  = dcdp_primitives( :dgt1_restr  , β, wp, zspace, Πp, ψspace)
    p4b  = dcdp_primitives( :Dgt0_restr  , β, wp, zspace, Πp, ψspace)
    p5   = dcdp_primitives( :one         , β, wp, zspace, Πp, ψspace)
    p6a  = dcdp_primitives( :dgt1        , β, wp, zspace, Πp, ψspace)
    p6b  = dcdp_primitives( :Dgt0        , β, wp, zspace, Πp, ψspace)

    σ = 0.25

    a = ShaleDrillingModel.STARTING_log_ogip # 0.561244
    b = ShaleDrillingModel.STARTING_σ_ψ      # 0.42165

    v3 = [-4.28566,       -5.45746,           -0.3, ]
    v4 = [-4.28566,       -4.71627, -5.80131, -0.3, ]
    v5 = [-4.28566, a, b, -5.45746,           -0.3, ]
    v6 = [-4.28566, a, b, -4.71627, -5.80131, -0.3, ]

    roy = 0.2
    geoid = 2

    @test check_flowgrad(v3, σ, p3 , geoid, roy)
    @test check_flowgrad(v4, σ, p4a, geoid, roy)
    @test check_flowgrad(v4, σ, p4b, geoid, roy)
    @test check_flowgrad(v5, σ, p5 , geoid, roy)
    @test check_flowgrad(v6, σ, p6a, geoid, roy)
    @test check_flowgrad(v6, σ, p6b, geoid, roy)
end
