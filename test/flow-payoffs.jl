
@testset "Flow gradients" begin
    problem = StaticDrillingPayoff(
        ShaleDrillingModel.DrillingRevenue(),
        ShaleDrillingModel.DrillingCost_constant(),
        ShaleDrillingModel.ExtensionCost_Constant()
    )

    let θ = fill(0.25, length(problem)),
        σ = 0.75
        ShaleDrillingModel.check_coef_length(problem, θ)
    end

    ShaleDrillingModel.showtypetree(AbstractPayoffFunction)

    println("")

    types_to_test = (
        DrillingCost_TimeFE(2008,2012),
        DrillingCost_TimeFE(2009,2011),
        DrillingCost_constant(),
        DrillingCost_dgt1(),
        # DrillingCost_TimeFE_rigrate(2008,2012),  # requires a different-sized state-space
        DrillingRevenue_WithTaxes(),
        DrillingRevenue(),
        ConstrainedDrillingRevenue_WithTaxes(),
        ConstrainedDrillingRevenue_WithTaxes(),
        ExtensionCost_Zero(),
        ExtensionCost_Constant(),
        ExtensionCost_ψ(),
        StaticDrillingPayoff(                       DrillingRevenue(), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()),
        UnconstrainedProblem( StaticDrillingPayoff( DrillingRevenue(), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()) ),
        ConstrainedProblem(   StaticDrillingPayoff( DrillingRevenue(), DrillingCost_TimeFE(2009,2011), ExtensionCost_Constant()) ),
        StaticDrillingPayoff(ConstrainedDrillingRevenue(), DrillingCost_constant(), ExtensionCost_Constant())
    )

    for f in types_to_test
        println("Testing fct $f")
        let z = (2.5,2010), ψ = 1.0, geoid = 4.5, roy = 0.25, σ = 0.75

            n = length(f)
            θ0 = rand(n)
            fd = zeros(Float64, n)
            g = zeros(Float64, n)

            for (d,i) in Iterators.product(0:2, 1:3)

                # test ∂f/∂θ
                Calculus.finite_difference!((thet) -> flow(f, thet, σ, wp, i, d, z, ψ, geoid, roy), θ0, fd, :central)
                ShaleDrillingModel.gradient!(f, θ0, g, σ, wp, i, d, z, ψ, geoid, roy)
                @test g ≈ fd

                # test ∂f/∂ψ
                fdpsi = Calculus.derivative((psi) -> flow(f, θ0, σ, wp, i, d, z, psi, geoid, roy), ψ)
                gpsi = flowdψ(f, θ0, σ, wp, i, d, z, ψ, geoid, roy)
                @test isapprox(fdpsi, gpsi, atol=1e-4)

                # test ∂f/∂σ
                fdsig = Calculus.derivative((sig) -> flow(f, θ0, sig, wp, i, d, z, ψ, geoid, roy), σ)
                gsig = flowdσ(f, θ0, σ, wp, i, d, z, ψ, geoid, roy)
                @test isapprox(fdsig, gsig, atol=1e-4)
            end
        end
    end
end



@testset "Old flow gradient tests" begin

    let zs_p   = (logp_space, ),
        zs_pc  = (logp_space, logc_space,),
        zs_py  = (logp_space,             2003:2012,),
        zs_pcy = (logp_space, logc_space, 2003:2012),
        roy = 0.225,
        geoid = 4.706,
        σ = 0.75,
        f1 = StaticDrillingPayoff(DrillingRevenue(), DrillingCost_constant(),                ExtensionCost_Constant()),
        f2 = StaticDrillingPayoff(DrillingRevenue(), DrillingCost_TimeFE(2009,2011),         ExtensionCost_Constant()),
        f3 = StaticDrillingPayoff(DrillingRevenue(), DrillingCost_TimeFE_rigrate(2008,2012), ExtensionCost_Constant())

        function testfun(FF::AbstractStaticPayoffs, zspace::Tuple)
            println("testing $FF")
            p = dcdp_primitives(FF, β, wp, zspace, Πp, ψspace)
            check_flowgrad(p, fill(0.3, length(FF)), σ, geoid, roy)
        end

        println("Testing a bunch of small flow gradients now...")

        @test testfun(f1, zs_p)
        @test testfun(f2, zs_py)
        @test testfun(f3, zs_pcy)
    end
end
