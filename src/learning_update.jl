export learningUpdate!, check_dΠψ

# ------------------------------ matrix updates -------------------------

# Information is not updated here, so we don't expect a transition
function _βΠψ!(P::AbstractMatrix, lrn::Union{NoLearn,PerfectInfo}, y1::StepRangeLen, y2::StepRangeLen, θp::Real, β::Real)
    (n = LinearAlgebra.checksquare(P)) == length(y1) == length(y2) || throw(DimensionMismatch())
    zero!(P)
    @inbounds for i in Base.OneTo(n)
        P[i,i] = β
    end
end

# information IS updated here... so we DO expect a transition
function _βΠψ!(P::AbstractMatrix, lrn::AbstractLearningType, y1::StepRangeLen, y2::StepRangeLen, θp::Real, β::Real)
    (n = LinearAlgebra.checksquare(P)) == length(y1) == length(y2) || throw(DimensionMismatch())
    ρ = _ρ(θp, lrn)
    Δ = 0.5 * step(y2)

    @inbounds for (j,yj) in enumerate(y2)
        if j == 1
            @. P[ :,j] = β * normcdf(_z(yj, y1, Δ, ρ))
        elseif j == n
            @. P[ :,j] = β * normccdf(_z(yj, y1, -Δ, ρ))
        else
            @. P[ :,j] = β * ( normcdf(_z(yj, y1, Δ, ρ)) - normcdf(_z(yj, y1, -Δ, ρ) ))
        end
    end
end

function _βΠψ(lrn::AbstractLearningType, y1::StepRangeLen{T}, θp::Real, β::Real) where {T}
    P = Matrix{T}(undef, length(y1), length(y1))
    _βΠψ!(P, lrn, y1, y1, θp, β)
    return P
end

_βΠψ!(P::AbstractMatrix, lrn::AbstractLearningType, y1::StepRangeLen, θp::Real, β::Real) = _βΠψ!(P, lrn, y1, y1, θp, β)
_βΠψ!(P::AbstractMatrix,                            y1::StepRangeLen, θp::Real, β::Real) = _βΠψ!(P, Learn(), y1, θp, β)

function _βΠψ(y1::StepRangeLen{T}, θp::Real, β::Real) where {T}
    P = Matrix{T}(undef, length(y1), length(y1))
    _βΠψ!(P, y1, y1, θp, β)
    return P
end

_βΠψdθρ!(P::AbstractMatrix, x::Learn,                y::StepRangeLen, θρ::Real, β::Real) = _βΠψdθρ!(P, y, θρ, β)
_βΠψdθρ!(P::AbstractMatrix, x::AbstractLearningType, y::StepRangeLen, θρ::Real, β::Real) = throw(error("Should not be calculating gradient with $x"))


function _βΠψdθρ!(P::AbstractMatrix, y::StepRangeLen, θρ::Real, β::Real)
    (n = LinearAlgebra.checksquare(P)) == length(y) || throw(DimensionMismatch())
    ρ = _ρ(θρ)
    dρdθρ = _dρdθρ(θρ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * _dπdρ(yj,y,Δ,ρ) * dρdθρ
        elseif j == n
            @. P[ :,j] = -β * _dπdρ(yj,y,-Δ,ρ) * dρdθρ
        else
            @. P[ :,j] = β * ( _dπdρ(yj,y,Δ,ρ) - _dπdρ(yj,y,-Δ,ρ) ) * dρdθρ
        end
    end
end

# ------------------------------ derivative check -------------------------

function check_dΠψ(θρ::Real, ψspace::StepRangeLen)

    Δ = 0.5 * step(ψspace)

    for yj in ψspace
        for y in ψspace
            fdσ = Calculus.derivative((sig) -> normcdf(_z(yj, y, Δ, _ρ(sig))), θρ)
            dσ  = _dρdθρ(θρ) * _dπdρ(yj, y, Δ, _ρ(θρ))
            abs(fdσ - dσ ) < 1e-7 || throw(error("bad σ grad at σ = $σ, ψ2 = $yj, ψ1 = $y"))
        end
    end
    return true
end

# ------------------------------ derivative check -------------------------

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
    f = flow(p)
    lrn = f.revenue.learn
    _βΠψ!(Πψtmp, lrn, ψspace, σ, p.β)
    A_mul_B_md!(EV2, Πψtmp, EV1, 2)

    if dograd
        # dubVtilde/dθ = du/dθ[:,:,:,2:dmaxp1] + β * Πψ ⊗ I * dEV/dθ[:,:,:,2:dmaxp1]
        A_mul_B_md!(dEV2, Πψtmp, dEV1, 2)

        # ∂EVtilde/∂σ[:,:,2:dmaxp1] = ∂u/∂σ[:,:,2:dmaxp1] + β * dΠψ/dσ ⊗ I * EV[:,:,2:dmaxp1]
        _βΠψdθρ!(Πψtmp, lrn, ψspace, σ, p.β)
        A_mul_B_md!(dEVσ2, Πψtmp, EV1, 2)
    end
end
