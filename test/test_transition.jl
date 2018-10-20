
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
