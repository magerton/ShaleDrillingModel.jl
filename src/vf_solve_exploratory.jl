export solve_vf_explore!

# must precede with running the following:
# learningUpdate!(ubV, dubV, dubV_σ, uex, duex, duexσ, EV, dEV, s2idx, βΠψ, βdΠψ, ψspace, vspace, σ, β)
# learningUpdate!(ubV, uex, EV, s2idx, βΠψ, ψspace, σ, β, v, h)

function solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θt::AbstractVector, σ::Real, dograd::Bool)

    EV       = evs.EV
    dEV      = evs.dEV
    dEVσ     = evs.dEVσ
    uex      = t.u
    duex     = t.du
    duexσ    = t.duσ
    ubVfull  = t.ubVfull
    dubVfull = t.dubVfull
    dubV_σ   = t.dubV_σ
    q        = t.q
    lse      = t.lse
    tmp      = t.tmp
    wp       = p.wp
    Πz       = p.Πz
    β        = p.β

    nz,nψ,nS = size(EV)
    nSexp, dmaxp1, nd = _nSexp(wp), exploratory_dmax(wp)+1, dmax(wp)+1

    (nz,nψ,dmaxp1) == size(uex)       || throw(DimensionMismatch())
    (nz,nψ,nd) == size(ubVfull)       || throw(DimensionMismatch())
    (nz,nz) == size(Πz)               || throw(DimensionMismatch())
    (nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)     == size(dEV)      || throw(DimensionMismatch())
        (nz,nψ,nSexp)     == size(dEVσ)     || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(duex)     || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(dubVfull) || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(dubV_σ)   || throw(DimensionMismatch())
    end

    # --------- VFIt --------------

    @views ubV =   ubVfull[:,:,  1:dmaxp1]
    @views dubV = dubVfull[:,:,:,1:dmaxp1]

    for i in explore_state_inds(wp)
        ip = action0(wp,i)
        idxs = sprime_idx(p,i)
        st = state(wp,i)

        fillflows!(ubV, flow, p, θt, σ, st, itype...)
        ubV[:,:,1]     .+= β .* EV[:,:,ip]
        ubV[:,:,2:end] .+= β .* EV[:,:,idx_lrn]

        # TODO: fillflows! for ubV, then ubV .+= β .* EV[:,:,actions()]
        # algorithm
        # (1) pin down EV[terminal]
        # (2) fill u
        # (3) form ubV[terminal-1] .= u[terminal-1] .+ β .* EV[terminal]
        # (4) EV[terminal-1] .= logsumexp(ubV)
        # (5) repeat 2-4

        # so it makes sense that size(u) = (1:z, 1:ψ, 1:d, 1:wp)
        # and size(du) = (nz,nψ,nθ,dmaxp1,nS)


        if dograd
            fillflows_grad!(dubV,   flowdθ, p, θt, σ, st, itype...)
            fillflows!(     dubV_σ, flowdσ, p, θt, σ, st, itype...)
            @views dubV[:,:,:,1] .+= β .* dEV[:,:,:,ip]
            @views dubV_σ[:,:,1] .+= β .* dEVσ[ :,:,ip]

            # this does EV0 & ∇EV0
            @views vfit!(EV[:,:,i], dEV[:,:,:,i], ubV, dubV, q, lse, tmp, Πz)

            # ∂EV/∂σ = I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂σ[zspace, ψspace, d]  )
            sumprod!(tmp, dubV_σ, q)
            @views A_mul_B_md!(dEVσ[:,:,i], Πz, tmp, 1)

        else
            @views vfit!(EV[:,:,i], ubV, lse, tmp, Πz)
        end
    end
end




#
