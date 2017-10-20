println("testing learning transition")

# ---------------- check on ∂Πψ/∂σ ------------------

ShaleDrillingModel.check_dπdσ(σv, ψspace, vspace)

let x2 = similar(tmpv.βΠψ),
    x1 = similar(tmpv.βΠψ),
    fd = similar(tmpv.βΠψ),
    d = tmpv.βdΠψ,
    v = vspace[3],
    h =  cbrt(eps(Float64))

    ShaleDrillingModel._fdβΠψ!(x2, ψspace, σv, β, v, h)
    ShaleDrillingModel._fdβΠψ!(x1, ψspace, σv, β, v, -h)
    ShaleDrillingModel._dβΠψ!(d, ψspace, σv, β, v)

    fd .= (x2 .- x1) ./ (2.0 .* h)
    # @show maximum(abs.(fd - d))
    @test fd ≈ d

    ShaleDrillingModel._βΠψ!(tmpv.βΠψ, ψspace, σv, 1.0)
    @test all(sum(tmpv.βΠψ, 2) .≈ 1.0)

    ShaleDrillingModel._fdβΠψ!(x2, ψspace, σv, 1.0, 1.0, cbrt(eps(Float64)))
    @test all(sum(x2, 2) .≈ 1.0)
end
