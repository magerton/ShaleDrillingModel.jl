export solve_vf_explore!

# must precede with running the following:
# learningUpdate!(ubV, dubV, dubV_σ, uex, duex, duexσ, EV, dEV, s2idx, βΠψ, βdΠψ, ψspace, vspace, σ, β)
# learningUpdate!(ubV, uex, EV, s2idx, βΠψ, ψspace, σ, β, v, h)


function solve_vf_explore!(
    EV::AbstractArray3     , dEV::AbstractArray4     , dEVσ::AbstractArray3 , # dEV_ψ::AbstractArray3 ,  # complete VF
    uex::AbstractArray3    , duex::AbstractArray4    , duexσ::AbstractArray3 , # duexψ::AbstractArray3 ,  # flow payoffs
    ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray3, # dubV_ψ::AbstractArray3,  # choice-specific VF
    q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix,                                        # temp vars
    wp::well_problem, Πz::AbstractMatrix, β::Real,                                                      # transitions, etc
    )

    nz,nψ,nS = size(EV)
    nSexp, dmaxp1, nd = _nSexp(wp), exploratory_dmax(wp)+1, dmax(wp)+1

    dograd = (length(dEV) > 0)

    (nz,nψ,dmaxp1) == size(uex)       || throw(DimensionMismatch())
    (nz,nψ,nd) == size(ubVfull)       || throw(DimensionMismatch())
    (nz,nz) == size(Πz)               || throw(DimensionMismatch())
    (nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)     == size(dEV)      || throw(DimensionMismatch())
        (nz,nψ,nSexp)     == size(dEVσ)    || throw(DimensionMismatch())
        # (nz,nψ,nSexp)     == size(dEV_ψ)    || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(duex)     || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(dubVfull) || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(dubV_σ)   || throw(DimensionMismatch())
        # (nz,nψ,dmaxp1)    == size(dubV_ψ)   || throw(DimensionMismatch())

        dubV = @view(dubVfull[:,:,:,1:dmaxp1])
    end

    # --------- VFIt --------------

    ubV = @view(ubVfull[:,:,1:dmaxp1])

    for i in explore_state_inds(wp)
        ip = action0(wp,i)
        @views ubV[:,:,1] .= β .* EV[:,:,ip]

        if dograd
            # TODO: assumes that u(0) = 0
            @views dubV[:,:,:,1] .= β .* dEV[:,:,:,ip]
            @views dubV_σ[:,:,1] .= β .* dEVσ[:,:,ip]
            # @views dubV_ψ[:,:,1] .= β .* dEV_ψ[:,:,ip]

            # this does EV0 & ∇EV0
            @views vfit!(EV[:,:,i], dEV[:,:,:,i], ubV, dubV, q, lse, tmp, Πz)

            # ∂EV/∂σ = I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂σ[zspace, ψspace, d]  )
            sumprod!(tmp, dubV_σ, q)
            @views A_mul_B_md!(dEVσ[:,:,i], Πz, tmp, 1)

            # # ∂EV/∂ψ = I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂ψ[zspace, ψspace, d]  )
            # sumprod!(tmp, dubV_ψ, q)
            # @views A_mul_B_md!(dEV_ψ[:,:,i], Πz, tmp, 1)
        else
            @views vfit!(EV[:,:,i], ubV, lse, tmp, Πz)
        end
    end
end



function solve_vf_explore!(EV::AbstractArray3{T}, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, β::Real) where {T}
    zeros4 = Array{T}(undef, 0,0,0,0)
    zeros3 = Array{T}(undef, 0,0,0)
    solve_vf_explore!(EV,     zeros4, zeros3, # zeros3,
                     uex,     zeros4, zeros3, # zeros3,
                     ubVfull, zeros4, zeros3, # zeros3,
                     zeros3, lse, tmp, wp, Πz, β
                     )
end

solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, ::Type{Val{true}})  = solve_vf_explore!(evs.EV, evs.dEV, evs.dEVσ, t.uex, t.duex, t.duexσ, t.ubVfull, t.dubVfull, t.dubV_σ, t.q, t.lse, t.tmp, p.wp, p.Πz, p.β)
solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, ::Type{Val{false}}) = solve_vf_explore!(evs.EV,                     t.uex,                  t.ubVfull,                            t.lse, t.tmp, p.wp, p.Πz, p.β)
solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, dograd::Bool=true)  = solve_vf_explore!(evs, t, p, Val{dograd})









#
