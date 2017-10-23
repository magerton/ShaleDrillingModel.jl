export learningUpdate!


function learningUpdate!(ubV::AbstractArray3, uex::AbstractArray3, EV::AbstractArray3, s2idx::AbstractVector, Πψtmp::AbstractMatrix, ψspace::StepRangeLen, σ::Real, β::Real)
    _βΠψ!(Πψtmp, ψspace, σ, β)
    @views A_mul_B_md!(ubV[:,:,2:end], Πψtmp, EV[:,:,s2idx], 2)
    @views ubV[:,:,2:end] .+= uex[:,:,2:end]
end


function learningUpdate!(
    ubV::AbstractArray3, dubV::AbstractArray4, dubV_σ::AbstractArray3, dubV_ψ::AbstractArray3,
    uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray3, duexψ::AbstractArray3,
    EV::AbstractArray3, dEV::AbstractArray4, s2idx::AbstractVector,
    Πψtmp::AbstractMatrix,
    ψspace::StepRangeLen, σ::Real, β::Real)

    dmaxp1 = length(s2idx)

    # update alternative-specific value functions for drilling 1+ wells & entering infill drilling regime
    ubV1    = @view(ubV[   :,:,  2:end])
    dubV1   = @view(dubV[  :,:,:,2:end])
    dubV_σ1 = @view(dubV_σ[:,:,  2:end])
    dubV_ψ1 = @view(dubV_ψ[:,:,  2:end])

    EV1     = @view(EV[    :,:,  s2idx])
    dEV1    = @view(dEV[   :,:,:,s2idx])

    # ----------- value --------------
    # ubVtilde = u[:,:,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(ubV1, Πψtmp, EV1, 2)
    @views ubV1 .+= uex[:,:,2:end]

    # ---------- gradient ----------------
    # dubVtilde/dθ = du/dθ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dθ[:,:,:,2:dmaxp1]
    A_mul_B_md!(dubV1, Πψtmp, dEV1, 2)
    @views dubV1 .+= duex[:,:,:,2:end]

    # ∂EVtilde/∂σ[:,:,2:dmaxp1] = ∂u/∂σ[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψdσ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(dubV_σ1, Πψtmp, EV1, 2)
    @views dubV_σ1 .+= duexσ[:,:,2:end]

    # ∂EVtilde/∂ψ[:,:,2:dmaxp1] = ∂u/∂ψ[:,:,2:dmaxp1] + β * dΠψ/dψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψdψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(dubV_ψ1, Πψtmp, EV1, 2)
    @views dubV_ψ1 .+= duexψ[:,:,2:end]
end

function learningUpdateψpeturb!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::T, h::T) where {T}
    dmaxp1 = exploratory_dmax(p.wp)+1
    s2idx = infill_state_idx_from_exploratory(p.wp)
    ubV  = @view( t.ubVfull[:,:,1:dmaxp1])
    learningUpdate!(ubV, t.uex, evs.EV, s2idx, t.Πψtmp, _ψspace(p,σ) .+ h, σ, p.β, h)
end


function learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, ::Type{Val{false}})
    dmaxp1 = exploratory_dmax(p.wp)+1
    s2idx = infill_state_idx_from_exploratory(p.wp)
    ubV  = @view(t.ubVfull[:,:,1:dmaxp1])
    learningUpdate!(ubV, t.uex, evs.EV, s2idx, t.Πψtmp, _ψspace(p,σ), σ, p.β)
end

# function learningUpdate!(
#     ubV::AbstractArray3, dubV::AbstractArray4, dubV_σ::AbstractArray3, dubV_ψ::AbstractArray3,
#     uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray3, duexψ::AbstractArray3,
#     EV::AbstractArray3, dEV::AbstractArray4, s2idx::AbstractVector,
#     Πψtmp::AbstractMatrix,
#     ψspace::StepRangeLen, σ::Real, β::Real)

function learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, ::Type{Val{true}})
    dmaxp1 = exploratory_dmax(p.wp)+1
    s2idx = infill_state_idx_from_exploratory(p.wp)
    ubV  = @view( t.ubVfull[:,:,1:dmaxp1])
    dubV = @view(t.dubVfull[:,:,:,1:dmaxp1])
    learningUpdate!(ubV, dubV, t.dubV_σ, t.dubV_ψ, t.uex, t.duex, t.duexσ, t.duexψ, evs.EV, evs.dEV, s2idx, t.Πψtmp, _ψspace(p,σ), σ, p.β)
end

learningUpdate!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, σ::Real, dograd::Bool=true)= learningUpdate!(evs, t, p, σ, Val{dograd})
