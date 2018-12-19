@testset "Flow gradients" begin

    p3   = dcdp_primitives( :one_restr   , β, wp, zspace, Πp, ψspace)
    p4a  = dcdp_primitives( :dgt1_restr  , β, wp, zspace, Πp, ψspace)
    p4b  = dcdp_primitives( :Dgt0_restr  , β, wp, zspace, Πp, ψspace)
    p5   = dcdp_primitives( :one         , β, wp, zspace, Πp, ψspace)

    p6a  = dcdp_primitives( :dgt1        , β, wp, zspace, Πp, ψspace)
    p6b  = dcdp_primitives( :Dgt0        , β, wp, zspace, Πp, ψspace)

    p5a = dcdp_primitives( :dgt1_ext_restr, β, wp, zspace, Πp, ψspace)
    p7a = dcdp_primitives( :dgt1_ext      , β, wp, zspace, Πp, ψspace)

    p5c = dcdp_primitives( :dgt1_d1_restr, β, wp, zspace, Πp, ψspace)
    p7c = dcdp_primitives( :dgt1_d1      , β, wp, zspace, Πp, ψspace)

    p5d = dcdp_primitives( :dgt1_cost_restr, β, wp, (logp_space, logc_space, ), Πpconly, ψspace)
    p7d = dcdp_primitives( :dgt1_cost      , β, wp, (logp_space, logc_space, ), Πpconly, ψspace)

    p6e = dcdp_primitives( :dgt1_cost_Dgt0_restr, β, wp, (logp_space, logc_space, ), Πpconly, ψspace)
    p8e = dcdp_primitives( :dgt1_cost_Dgt0      , β, wp, (logp_space, logc_space, ), Πpconly, ψspace)

    p6f = dcdp_primitives( :dgt1_pricecost_restr, β, wp, (logp_space, logc_space, ), Πpconly, ψspace)
    p8f = dcdp_primitives( :dgt1_pricecost      , β, wp, (logp_space, logc_space, ), Πpconly, ψspace)
    p7f = dcdp_primitives( :dgt1_pricebreak_restr, β, wp, (logp_space, logc_space, ), Πpconly, ψspace)
    p9f = dcdp_primitives( :dgt1_pricebreak      , β, wp, (logp_space, logc_space, ), Πpconly, ψspace)

    σ = 0.25

    a = STARTING_log_ogip # 0.561244
    b = STARTING_σ_ψ      # 0.42165

    v3 = [-4.28566,       -5.45746,           -0.3, ]
    v4 = [-4.28566,       -4.71627, -5.80131, -0.3, ]
    v5 = [-4.28566, a, b, -5.45746,           -0.3, ]
    v6 = [-4.28566, a, b, -4.71627, -5.80131, -0.3, ]

    v5a = [-4.28566,       -4.71627, -5.80131, -0.2, -0.2, ]
    v7a = [-4.28566, a, b, -4.71627, -5.80131, -0.2, -0.2, ]

    v6e = [-4.28566,       -4.71627, -5.80131, -0.2, -0.2, -1.0,]
    v8e = [-4.28566, a, b, -4.71627, -5.80131, -0.2, -0.2, -1.0,]

    v6f = [-4.28566,       1.0, -2.71627, -1.80131, -0.7, -0.2, ]
    v8f = [-4.28566, a, b, 1.0, -2.71627, -1.80131, -0.7, -0.2, ]

    v7f = [-4.28566,       1.0, 1.0, -2.71627, -1.80131, -0.7, -0.2, ]
    v9f = [-4.28566, a, b, 1.0, 1.0, -2.71627, -1.80131, -0.7, -0.2, ]

    roy = 0.225    # median royalty
    geoid = 4.706  # median geology

    @test check_flowgrad(v7f, σ, p7f, geoid, roy)
    @test check_flowgrad(v9f, σ, p9f, geoid, roy)

    @test check_flowgrad(v6f, σ, p6f, geoid, roy)
    @test check_flowgrad(v8f, σ, p8f, geoid, roy)

    @test check_flowgrad(v6e, σ, p6e, geoid, roy)
    @test check_flowgrad(v8e, σ, p8e, geoid, roy)

    @test check_flowgrad(v5a, σ, p5d, geoid, roy)
    @test check_flowgrad(v7a, σ, p7d, geoid, roy)

    @test check_flowgrad(v5a, σ, p5c, geoid, roy)
    @test check_flowgrad(v7a, σ, p7c, geoid, roy)

    @test check_flowgrad(v5a, σ, p5a, geoid, roy)
    @test check_flowgrad(v7a, σ, p7a, geoid, roy)

    @test check_flowgrad(v3, σ, p3 , geoid, roy)
    @test check_flowgrad(v4, σ, p4a, geoid, roy)
    @test check_flowgrad(v4, σ, p4b, geoid, roy)
    @test check_flowgrad(v5, σ, p5 , geoid, roy)
    @test check_flowgrad(v6, σ, p6a, geoid, roy)
    @test check_flowgrad(v6, σ, p6b, geoid, roy)
end
