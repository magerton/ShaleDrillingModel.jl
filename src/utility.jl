export fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, st::Tuple, typestuff::Real...) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = flowdθ(FF, θ, σ, z, st[end-1], k, st[end], typestuff...)
    end
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, h::Real) = k
tweak(ψ::Real   , h::Real) = ψ + h

function tweakstate(st::Tuple,  h::Real)
    h == zero(h) && return st
    return tweak(st[end-1],h), st[end]
end

function flowfdψ(FF::Type, θ::AbstractVector,  σ::Real, z::Tuple, st::Tuple, d1::Integer, Dgt0::Bool, sgnext::Bool, τrem::Integer, h::Real, itype::Real...)
    flow(FF, θ, σ, z, tweakstate(st, h)..., d1, Dgt0, sgnext, τrem, itype...)
end

# ------------------------ check flow grad --------------

function check_flowgrad(FF::Type, θ::AbstractVector{T}, σ::T, zspace::Tuple, ψspace::AbstractRange, wp::well_problem, itype::Real...) where {T}
    K = length(θ)
    dx = Vector{T}(undef, K)
    fdx = similar(dx)

    zpdct = Base.product(zspace...)
    pdct = Base.product(zpdct, ψspace, 0:dmax(wp))
    statelist = (state(10,5,0,0), state(0,5,0,0), state(-1,4,0,0), state(-1,0,0,0), state(-1,-1,1,0), state(-1,-1,1,1),)

    for yy in pdct
        # Loop over deterministic states... (d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Integer, )
        for st in statelist
            u(θ) = flow(FF, θ, σ, yy...,            stateinfo(st)..., itype...)
            duθ!(dx,    FF, θ, σ, yy[1], yy[2:end], stateinfo(st)..., itype...)
            Calculus.finite_difference!(u, θ, fdx, :central)
            if !(fdx ≈ dx)
                @warn "FF = $FF. Bad θ diff at $yy, $st. du=$dx and fd = $dxfd"
                return false
            end
        end

        # check σ
        for st in statelist[1:1]
            dσ = flowdσ(FF, θ, σ, yy..., itype...)
            fdσ = Calculus.derivative((σh::Real) ->  flow(FF, θ, σh, yy..., stateinfo(st)..., itype...), σ, :central)
            if !(dσ ≈ fdσ) && !isapprox(dσ,fdσ, atol= 1e-7)
                @warn "Bad σ diff at (z,st,) = $((z,st,)). duσ = $dσ and fd = $fdσ"
                return false
            end
        end

        # check ψ
        for st in statelist[1:4]
            dψ = flowdψ(FF, θ, σ, yy..., _sgnext(st), itype...)
            fdψ = Calculus.derivative((h::Real) -> flowfdψ(FF, θ, σ, yy[1], yy[2:end], stateinfo(st)..., h, itype...), 0.0, :central)

            if !(dψ ≈ fdψ) && !isapprox(dψ, fdψ, atol=1e-7)
                @warn "Bad ψ diff at (z,st,) = $((z,st,)). duψ = $dψ and fdψ = $fdψ"
                return false
            end
        end
    end
    return true
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, θ, σ, p.zspace, p.ψspace, p.wp, itype...)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, _θt(θ, p), _σv(θ), p, itype...)

# ------------------------ fill flows --------------

function fillflows!(X::AbstractArray, f::Function, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, st::state, itype::Real...) where {FF}
    zpdct = Base.product(p.zspace...)
    pdct = Base.product(zpdct, p.ψspace, 0:dmax(p))
    size(X) == size(pdct) || throw(DimensionMismatch())
    @inbounds for (i,z,) in enumerate(pdct)
        X[i] = f(FF, θ, σ, z..., stateinfo(st)..., itype...)
    end
end

function fillflows_grad!(X::AbstractArray, f::Function, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, st::state, itype::Real...) where {FF}
    zpdct = Base.product(p.zspace...)
    pdct = Base.product(zpdct, p.ψspace, Base.OneTo(length(θ)), 0:dmax(p))
    size(X) == size(pdct) || throw(DimensionMismatch())
    @inbounds for (i,z,) in enumerate(pdct)
        X[i] = f(FF, θ, σ, z..., stateinfo(st)..., itype...)
    end
end
