@testset "learning transition small functions" begin
    for x0 ∈ (0.25, 0.5, 0.75,)
        vspace = range(-2.0, stop=2.0, length=5)
        @test Calculus.derivative(_ρ, x0) ≈ ShaleDrillingModel._dρdθρ(x0)
        @test Calculus.derivative(_ρ, x0) ≈ ShaleDrillingModel._dρdσ(x0)

        for u in vspace
            for v in vspace
                @test Calculus.derivative((z) -> _ψ1(u,v,_ρ(z)), x0) ≈ _dψ1dθρ(u,v,_ρ(x0), x0)
                @test Calculus.derivative((z) -> _ψ1(u,v,z), x0) ≈ ShaleDrillingModel._dψ1dρ(u,v,x0)
            end
        end
    end
end


@testset "Frechet derivative of ψ transition matrix" begin

    dpi = similar(tmpv.Πψtmp)
    fdpi = similar(tmpv.Πψtmp)
    h = peturb(σv)
    σ1 = σv - h
    σ2 = σv + h
    hh = σ2 - σ1

    ShaleDrillingModel._βΠψ!(dpi, ψspace, σ1, β)
    ShaleDrillingModel._βΠψ!(fdpi, ψspace, σ2, β)
    fdpi .-= dpi
    fdpi ./= hh
    ShaleDrillingModel._βΠψdθρ!(dpi, ψspace, σv, β)

    @views maxv, idx = findmax(abs.(dpi .- fdpi))
    sub = CartesianIndices(fdpi)[idx]
    @show "worst value is $maxv at $sub for β dΠ/dσ"

    @test 0.0 < maxv < 1e-5
    @test fdpi ≈ dpi

end
