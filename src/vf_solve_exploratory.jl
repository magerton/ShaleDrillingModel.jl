export solve_vf_explore!

# must precede with running the following:
# learningUpdate!(ubV, dubV, dubV_σ, uex, duex, duexσ, EV, dEV, s2idx, βΠψ, βdΠψ, ψspace, vspace, σ, β)
# learningUpdate!(ubV, uex, EV, s2idx, βΠψ, ψspace, σ, β, v, h)

function solve_vf_explore!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θt::AbstractVector, σ::Real, dograd::Bool, itype::Tuple; maxit0::Integer=40, maxit1::Integer=20, vftol::Real=1e-9)

    EV         = evs.EV
    dEV        = evs.dEV
    dEVσ       = evs.dEVσ
    wp         = p.wp
    Πz         = p.Πz
    β          = p.β

    nz,nψ,nS = size(EV)
    nSexp, dmaxp1, nd = _nSexp(wp), _dmax(wp)+1, _dmax(wp)+1

    (nz,nψ,nd) == size(t.ubVfull)         || throw(DimensionMismatch())
    (nz,nz) == size(Πz)                   || throw(DimensionMismatch())
    (nz,nψ) == size(t.lse) == size(t.tmp) || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)     == size(dEV)        || throw(DimensionMismatch())
        (nz,nψ,nSexp)     == size(dEVσ)       || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(t.q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(t.dubVfull) || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(t.dubV_σ)   || throw(DimensionMismatch())
    end

    # --------- VFIt --------------

    # Views of ubV so we can efficiently access them
    @views ubV0    = t.ubVfull[:,:,  1]
    @views ubV1    = t.ubVfull[:,:,  2:dmaxp1]
    @views ubV     = t.ubVfull[:,:,  1:dmaxp1]

    @views dubV0   = t.dubVfull[:,:,:,1]
    @views dubV1   = t.dubVfull[:,:,:,2:dmaxp1]
    @views dubV    = t.dubVfull[:,:,:,1:dmaxp1]

    @views dubV_σ0 = t.dubV_σ[  :,:,  1]
    @views dubV_σ1 = t.dubV_σ[  :,:,  2:dmaxp1]
    @views dubV_σ  = t.dubV_σ[  :,:,  1:dmaxp1]

    exp2lrn = exploratory_learning(wp)
    @views βEV1   =  EV[ :,:,  exp2lrn]
    @views βdEV1  = dEV[ :,:,:,exp2lrn]
    @views βdEVσ1 = dEVσ[:,:,  exp2lrn]

    for i in ind_exp(wp)
        ip = sprime(wp,i,0)
        horzn = _horizon(wp,i)

        @views EV0 = EV[:,:,ip]
        @views dEV0 = dEV[:,:,:,ip]
        @views dEVσ1 = dEVσ[ :,:,ip]

        # compute u + βEV(d) ∀ d ∈ actionspace(wp,i)
        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        ubV0 .+= β .* EV0
        ubV1 .+= βEV1 # β already baked in

        if dograd
            fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
            fillflows!(   dubV_σ, flowdσ, p, θt, σ, i, itype...)
            dubV0   .+= β .* dEV0
            dubV1   .+= βdEV1  # β already baked in
            dubV_σ0 .+= β .* dEVσ1
            dubV_σ1 .+= βdEVσ1 # β already baked in
        end

        if horzn == :Finite
            if dograd
                @views vfit!(EV[:,:,i], dEV[:,:,:,i], dEVσ[:,:,i], t, p)
            else
                @views vfit!(EV[:,:,i], t, p)
            end

        elseif horzn == :Infinite
            converged, iter, bnds =  solve_inf_vfit_pfit!(EV0, t, p; vftol=vftol, maxit0=maxit0, maxit1=maxit1)
            converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds. θt = $θt, σ = $σ"

            if dograd
                # TODO: only allows 0-payoff if no action
                ubV[:,:,1] .= β .* EV0
                gradinf!(dEV0, dEVσ1, t, p, true)
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
end
