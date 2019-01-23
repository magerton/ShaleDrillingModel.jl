export solve_vf_explore!

# must precede with running the following:
# learningUpdate!(ubV, dubV, dubV_σ, uex, duex, duexσ, EV, dEV, s2idx, βΠψ, βdΠψ, ψspace, vspace, σ, β)
# learningUpdate!(ubV, uex, EV, s2idx, βΠψ, ψspace, σ, β, v, h)

function solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θt::AbstractVector, σ::Real, dograd::Bool, itype::Tuple)

    EV       = evs.EV
    dEV      = evs.dEV
    dEVσ     = evs.dEVσ
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
    nSexp, dmaxp1, nd = _nSexp(wp), _dmax(wp)+1, _dmax(wp)+1

    (nz,nψ,nd) == size(ubVfull)       || throw(DimensionMismatch())
    (nz,nz) == size(Πz)               || throw(DimensionMismatch())
    (nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)     == size(dEV)      || throw(DimensionMismatch())
        (nz,nψ,nSexp)     == size(dEVσ)     || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(dubVfull) || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(dubV_σ)   || throw(DimensionMismatch())
    end

    # --------- VFIt --------------

    # Views of ubV so we can efficiently access them
    @views ubV0    =  ubVfull[:,:,  1]
    @views ubV1    =  ubVfull[:,:,  2:dmaxp1]
    @views dubV0   = dubVfull[:,:,:,1]
    @views dubV1   = dubVfull[:,:,:,2:dmaxp1]
    @views dubV_σ0 = dubV_σ[  :,:,  1]
    @views dubV_σ1 = dubV_σ[  :,:,  2:dmaxp1]

    exp2lrn = exploratory_learning(wp)
    @views βEV1   =  EV[ :,:,  exp2lrn]
    @views βdEV1  = dEV[ :,:,:,exp2lrn]
    @views βdEVσ1 = dEVσ[:,:,  exp2lrn]

    for i in ind_exp(wp)
        ip = sprime(wp,i,0)

        @views EV0 = EV[:,:,ip]

        # compute u + βEV(d) ∀ d ∈ actionspace(wp,i)
        fillflows!(ubVfull, flow, p, θt, σ, i, itype...)
        ubV0 .+= β .* EV0
        ubV1 .+= βEV1 # β already baked in

        if dograd
            @views dEV0 = dEV[:,:,:,ip]
            @views dEVσ1 = dEVσ[ :,:,ip]
            fillflows_grad!(dubVfull, flowdθ, p, θt, σ, i, itype...)
            fillflows!(       dubV_σ, flowdσ, p, θt, σ, i, itype...)
            dubV0   .+= β .* dEV0
            dubV1   .+= βdEV1  # β already baked in
            dubV_σ0 .+= β .* dEVσ1
            dubV_σ1 .+= βdEVσ1 # β already baked in

            # this does EV0 & ∇EV0
            @views vfit!(EV[:,:,i], dEV[:,:,:,i], ubVfull, dubVfull, q, lse, tmp, Πz)

            # ∂EV/∂σ = I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂σ[zspace, ψspace, d]  )
            sumprod!(tmp, dubV_σ, q)
            @views A_mul_B_md!(dEVσ[:,:,i], Πz, tmp, 1)
        else
            @views vfit!(EV[:,:,i], ubVfull, lse, tmp, Πz)
        end
    end
end




#
