export learningUpdate!

function learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, dograd::Bool)

    Πψtmp  = t.Πψtmp
    ψspace = _ψspace(p)

    d2plus = 2:_dmax(p.wp)+1  # TODO: would be good to make this more general.
    lrn2inf = inf_fm_lrn(p.wp)
    exp2lrn = exploratory_learning(p.wp)

    @views EV1   =  evs.EV[:,:,  lrn2inf]
    @views dEV1  = evs.dEV[:,:,:,lrn2inf]

    @views EV2   =  evs.EV[ :,:,  exp2lrn]
    @views dEV2  = evs.dEV[ :,:,:,exp2lrn]
    @views dEVσ2 = evs.dEVσ[:,:  ,exp2lrn]

    # ubVtilde = u[:,:,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψ!(Πψtmp, ψspace, σ, p.β)
    A_mul_B_md!(EV2, Πψtmp, EV1, 2)

    if dograd
        # dubVtilde/dθ = du/dθ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dθ[:,:,:,2:dmaxp1]
        A_mul_B_md!(dEV2, Πψtmp, dEV1, 2)

        # ∂EVtilde/∂σ[:,:,2:dmaxp1] = ∂u/∂σ[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
        _βΠψdθρ!(Πψtmp, ψspace, σ, p.β)
        A_mul_B_md!(dEVσ2, Πψtmp, EV1, 2)
    end
end
