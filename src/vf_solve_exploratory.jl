export solve_vf_explore!

# must precede with running the following:
# learningUpdate!(ubV, dubV, dubV_σ, uex, duex, duexσ, EV, dEV, s2idx, βΠψ, βdΠψ, ψspace, vspace, σ, β)
# learningUpdate!(ubV, uex, EV, s2idx, βΠψ, ψspace, σ, β, v, h)


function solve_vf_explore!(
    EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4,                          # complete VF
    uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray4,                        # flow payoffs
    ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray4,               # choice-specific VF
    q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix,                             # temp vars
    wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix, β::Real, # transitions, etc
    )

    nz,nψ,nS = size(EV)
    nSex, dmaxp1, nd = length(explore_state_inds(wp)), exploratory_dmax(wp)+1, dmax(wp)+1
    s2idx = infill_state_idx_from_exploratory(wp)

    dograd = (length(dEV) > 0)

    (nz,nψ,dmaxp1) == size(uex) || throw(DimensionMismatch())
    (nz,nψ,nd) == size(ubVfull) || throw(DimensionMismatch())
    (nz,nz) == size(Πz)         || throw(DimensionMismatch())
    (nψ,nψ) == size(βΠψ)        || throw(DimensionMismatch())
    (nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())

    if dograd
        nθ, nv = size(dEV,3), size(dEV_σ,3)
        (nz,nψ,nθ,nS)     == size(dEV)      || throw(DimensionMismatch())
        (nz,nψ,nv,nSex+1) == size(dEV_σ)    || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(duex)     || throw(DimensionMismatch())
        (nψ,nψ)           == size(βdΠψ)     || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(dubVfull) || throw(DimensionMismatch())

        dubV = @view(dubVfull[:,:,:,1:dmaxp1])
    end

    ubV = @view(ubVfull[:,:,1:dmaxp1])

    # --------- VFIt --------------

    for i in explore_state_inds(wp)
        ip = action0(wp,i)

        EV0 = @view(EV[:,:,i])
        ubV[:,:,1] .= β .* @view(EV[:,:,ip])

        if dograd
            dEV0 = @view(dEV[:,:,:,i])
            dEV1 = @view(dEV[:,:,:,ip])
            dEV0_σ = @view(dEV_σ[:,:,:,i])
            dEV1_σ = @view(dEV_σ[:,:,:,min(ip,nSex+1)])  # because we don't use regime2 but need a TVC
            sumdubV_σ = @view(dubV_σ[:,:,:,1])

            # TODO: assumes that u(0) = 0
            dubV[:,:,:,1] .= β .* dEV1
            dubV_σ[:,:,:,1] .= β .* dEV1_σ

            # this does EV0 & ∇EV0
            vfit!(EV0, dEV0, ubV, dubV, q, lse, tmp, Πz)

            # ∂EV/∂σ = I ⊗ I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂σ[zspace, ψspace, vspace, d]  )
            sumprod!(sumdubV_σ, dubV_σ, q)
            A_mul_B_md!(dEV0_σ, Πz, sumdubV_σ, 1)
        else
            vfit!(EV0, ubV, lse, tmp, Πz)
        end
    end
end


function solve_vf_explore!(EV::AbstractArray3{T}, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, β::Real) where {T}
    zeros4 = Array{T}(0,0,0,0)
    zeros3 = Array{T}(0,0,0)
    zeros2 = Array{T}(0,0)
    solve_vf_explore!(EV, zeros4, zeros4, uex, zeros4, zeros4, ubVfull, zeros4, zeros4, zeros3, lse, tmp, wp, Πz, βΠψ, zeros2, β)
end


solve_vf_explore!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives) = solve_vf_explore!(e.EV, e.dEV, e.dEV_σ, t.uex, t.duex, t.duexσ, t.ubVfull, t.dubVfull, t.dubV_σ, t.q, t.lse, t.tmp, p.wp, p.Πz, t.βΠψ, t.βdΠψ, β)









#
