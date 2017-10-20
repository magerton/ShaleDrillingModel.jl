export learningUpdate!

function learningUpdate!(ubV::AbstractArray3, uex::AbstractArray3, EV::AbstractArray3, s2idx::AbstractVector, βΠψ::AbstractMatrix, ψspace::StepRangeLen, σ::T, β::Real, v::Real=zero(T), h::T=zero(T)) where {T}
    ubV1 = @view(ubV[:,:,2:end])
    EV1 = @view(EV[:,:,s2idx])
    uex1 = @view(uex[:,:,2:end])

    if h == zero(T)
        _βΠψ!(βΠψ, ψspace, σ, β)
    else
        _fdβΠψ!(βΠψ, ψspace, σ, β, v, h)
    end

    A_mul_B_md!(ubV1, βΠψ, EV1, 2)
    ubV[:,:,2:end] .+= uex1
end


function learningUpdate!(
    ubV::AbstractArray3, dubV::AbstractArray4, dubV_σ::AbstractArray4,
    uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray4,
    EV::AbstractArray3, dEV::AbstractArray4, s2idx::AbstractVector,
    βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix,
    ψspace::StepRangeLen, vspace::AbstractVector, σ::T, β::Real,
    dograd::Bool=true, v::Real=zero(T), h::T=zero(T)
    ) where {T}

    length(vspace) == size(dubV_σ, 3) || throw(DimensionMismatch())

    ubV1    = @view(ubV[   :,:,  2:end])
    EV1     = @view(EV[    :,:,  s2idx])
    uex1    = @view(uex[   :,:,  2:end])

    if h == zero(T)
        _βΠψ!(βΠψ, ψspace, σ, β)
    else
        _fdβΠψ!(βΠψ, ψspace, σ, β, v, h)
    end

    # ubVtilde = u[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
    A_mul_B_md!(ubV1, βΠψ, EV1, 2)
    ubV[:,:,2:end] .+= uex1

    if dograd
        dubV1   = @view(dubV[  :,:,:,2:end])
        dubV_σ1 = @view(dubV_σ[:,:,:,2:end])
        dEV1    = @view(dEV[   :,:,:,s2idx])
        duex1   = @view(duex[  :,:,:,2:end])
        duex_σ1 = @view(duexσ[ :,:,:,2:end])

        # dubVtilde/dσ = du/dσ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dσ[:,:,:,2:dmaxp1]
        A_mul_B_md!(dubV1, βΠψ, dEV1, 2)
        dubV1 .+= duex1

        # ∂EVtilde/∂σ[:,:,v,2:dmaxp1] = ∂u/∂σ[:,:,v,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]  ∀  v ∈ vspace
        for d in 1:size(dubV1,4)
            EV1d = @view(EV1[:,:,d])
            for (k, v) in enumerate(vspace)
                dubV_σ1k = @view(dubV_σ1[:,:,k,d])
                _dβΠψ!(βdΠψ, ψspace, σ, β, v)
                A_mul_B_md!(dubV_σ1k, βdΠψ, EV1d, 2)
            end
        end
        dubV_σ[:,:,:,2:end] .+= duex_σ1
    end
end



function learningUpdate!(t::dcdp_tmpvars, e::dcdp_Emax, p::dcdp_primitives, σ::T, dograd::Bool=true, v::Real=zero(T), h::T=zero(T)) where {T}
    dmaxp1 = exploratory_dmax(p.wp)+1
    s2idx = infill_state_idx_from_exploratory(p.wp)
    ubV  = @view( t.ubVfull[:,:,1:dmaxp1])

    if dograd
        dubV = @view(t.dubVfull[:,:,:,1:dmaxp1])
        learningUpdate!(ubV, dubV, t.dubV_σ, t.uex, t.duex, t.duexσ, e.EV, e.dEV, s2idx, t.βΠψ, t.βdΠψ, _ψspace(p,σ), _vspace(p), σ, p.β)
    else
        learningUpdate!(ubV, t.uex, e.EV, s2idx, t.βΠψ, _ψspace(p,σ), σ, p.β, v, h)
    end
end
