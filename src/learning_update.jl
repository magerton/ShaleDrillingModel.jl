export learningUpdate!


function learningUpdate!(ubV::AbstractArray3, uex::AbstractArray3, EV::AbstractArray3, wp::well_problem, Πψtmp::AbstractMatrix, ψspace::StepRangeLen, σ::Real, β::Real)
    d2plus = 2:dmax(wp)+1  # TODO: would be good to make this more general.
    exp2lrn = exploratory_learning(wp)
    lrn2inf = inf_fm_lrn(wp)

    _βΠψ!(Πψtmp, ψspace, σ, β)
    @views A_mul_B_md!(ubV[:,:,d2plus], Πψtmp, EV[:,:,lrn2inf], 2)
    @views EV[:,:,exp2lrn] .=  ubV[:,:,d2plus]
    @views ubV[:,:,d2plus] .+= uex[:,:,d2plus]
end


function learningUpdate!(
    ubV::AbstractArray3, dubV::AbstractArray4, dubV_σ::AbstractArray3, dubV_ψ::AbstractArray3,
    uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray3, duexψ::AbstractArray3,
    EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray3, dEV_ψ::AbstractArray3,
    wp::well_problem,
    Πψtmp::AbstractMatrix,
    ψspace::StepRangeLen, σ::Real, β::Real)

    d2plus = 2:dmax(wp)+1  # TODO: would be good to make this more general.
    lrn2inf = inf_fm_lrn(wp)
    exp2lrn = exploratory_learning(wp)

    # update alternative-specific value functions for drilling 1+ wells & entering infill drilling regime
    ubV1    = @view(ubV[   :,:,  d2plus])
    dubV1   = @view(dubV[  :,:,:,d2plus])
    dubV_σ1 = @view(dubV_σ[:,:,  d2plus])
    dubV_ψ1 = @view(dubV_ψ[:,:,  d2plus])

    EV1     = @view(EV[ :,:,  lrn2inf])
    dEV1    = @view(dEV[:,:,:,lrn2inf])

    # ----------- value --------------
    # ubVtilde = u[:,:,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(ubV1, Πψtmp, EV1, 2)
    EV[:,:,exp2lrn] .= ubV1
    @views ubV1 .+= uex[:,:,d2plus]

    # ---------- gradient ----------------
    # dubVtilde/dθ = du/dθ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dθ[:,:,:,2:dmaxp1]
    A_mul_B_md!(dubV1, Πψtmp, dEV1, 2)
    dEV[:,:,:,exp2lrn] .= dubV1
    @views dubV1 .+= duex[:,:,:,d2plus]

    # ∂EVtilde/∂σ[:,:,2:dmaxp1] = ∂u/∂σ[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψdσ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(dubV_σ1, Πψtmp, EV1, 2)
    dEV_σ[:,:,exp2lrn] .= dubV_σ1
    @views dubV_σ1 .+= duexσ[:,:,d2plus]

    # ∂EVtilde/∂ψ[:,:,2:dmaxp1] = ∂u/∂ψ[:,:,2:dmaxp1] + β * dΠψ/dψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψdψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(dubV_ψ1, Πψtmp, EV1, 2)
    dEV_ψ[:,:,exp2lrn] .= dubV_ψ1
    @views dubV_ψ1 .+= duexψ[:,:,d2plus]
end


learningUpdateψpeturb!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::T, h::T) where {T} = learningUpdate!(t.ubVfull,                                 t.uex,                           evs.EV,                                p.wp, t.Πψtmp, _ψspace(p,σ).+h, σ, p.β, h)
learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, ::Type{Val{false}}) = learningUpdate!(t.ubVfull,                                 t.uex,                           evs.EV,                                p.wp, t.Πψtmp, _ψspace(p,σ),    σ, p.β)
learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, ::Type{Val{true}})  = learningUpdate!(t.ubVfull, t.dubVfull, t.dubV_σ, t.dubV_ψ, t.uex, t.duex, t.duexσ, t.duexψ, evs.EV, evs.dEV, evs.dEV_σ, evs.dEV_ψ, p.wp, t.Πψtmp, _ψspace(p,σ),    σ, p.β)
learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, dograd::Bool=true)  = learningUpdate!(evs, t, p, σ, Val{dograd})





#
