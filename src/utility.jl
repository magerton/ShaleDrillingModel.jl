export fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, FF::Type, wp::AbstractUnitProblem, θ::AbstractVector{T}, σ::T, z::Tuple, stinfo::Tuple, typestuff::Real...) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    stinfo[end]
    @inbounds for k = 1:K
        du[k] = flowdθ(FF, wp, θ, σ, z, stinfo[end-1], k, stinfo[end], typestuff...)
    end
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, h::Real) = k
tweak(ψ::Real   , h::Real) = ψ + h

function tweakstate(stinfo::Tuple,  h::Real)
    h == zero(h) && return st
    return tweak(stinfo[end-1],h), stinfo[end]
end

function flowfdψ(FF::Type, wp::AbstractUnitProblem, sidx::Integer, θ::AbstractVector, σ::Real, h::Real, z::Tuple, ψ::T, d::Integer, itype::Real...) where {T}
    flow(FF, wp, sidx, θ, σ, z, ψ+h, d, itype...)
end

# ------------------------ check flow grad --------------

function check_flowgrad(FF::Type, θ::AbstractVector{T}, σ::T, zspace::Tuple, ψspace::AbstractRange, wp::AbstractUnitProblem, itype::Real...) where {T}
    K = length(θ)
    dx = Vector{T}(undef, K)
    fdx = similar(dx)

    zpdct = Base.product(zspace...)
    pdct = Base.product(zpdct, ψspace, 0:_dmax(wp))

    statelist = state_space_vector(wp)

    for zψd in pdct
        # Loop over deterministic states... (d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Integer, )
        for sidx ∈ 1:ShaleDrillingModel._nS(wp)
            u(θ) = flow(FF, wp, sidx, θ, σ, zψd..., itype...)
            Calculus.finite_difference!(u, θ, fdx, :central)
            @inbounds for k ∈ 1:K
                dx[k] = flowdθ(FF, wp, sidx, θ, σ, zψd[1], zψd[2], k, zψd[3], itype...)
            end
            if !(fdx ≈ dx)
                @warn "FF = $FF. Bad θ diff at $yy, $st. du=$dx and fd = $dxfd"
                return false
            end
        end

        # check σ
        for sidx ∈ 1:2
            dσ = flowdσ(FF, wp, sidx, θ, σ, zψd..., itype...)
            fdσ = Calculus.derivative((σh::Real) -> flow(FF, wp, sidx, θ, σh, zψd..., itype...), σ, :central)
            if !(dσ ≈ fdσ) && !isapprox(dσ, fdσ, atol= 1e-7)
                @warn "Bad σ diff at zψd = $(zψd). duσ = $dσ and fd = $fdσ"
                return false
            end
        end

        # check ψ
        for sidx in 1:4
            dψ = flowdψ(FF, wp, sidx, θ, σ, zψd..., itype...)
            fdψ = Calculus.derivative((h::Real) -> flowfdψ(FF, wp, sidx, θ, σ, h, zψd..., itype...), 0.0, :central)
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

function fillflows!(X::AbstractArray, f::Function, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, sidx::Integer, itype::Real...) where {FF}
    zpdct = Base.product(p.zspace...)
    idxd = actionspace(p.wp, sidx)
    pdct = Base.product(zpdct, p.ψspace, idxd)
    length(X) == length(pdct) || throw(DimensionMismatch("state[$sidx] is $(state(p.wp,sidx)). size(X) = $(size(X)) != size(pdct) = $(size(pdct))"))
    @inbounds for (i,zψd,) in enumerate(pdct)
        X[i] = f(FF, p.wp, sidx, θ, σ, zψd..., itype...)
    end
end

function fillflows_grad!(X::AbstractArray, f::Function, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, sidx::Integer, itype::Real...) where {FF}
    zpdct = Base.product(p.zspace...)
    idxd = 0:size(X, ndims(X))-1
    pdct = Base.product(zpdct, p.ψspace, Base.OneTo(length(θ)), idxd)
    length(X) == length(pdct) || throw(DimensionMismatch("state[$sidx] is $(state(p.wp,sidx)). size(X) = $(size(X)) != size(pdct) = $(size(pdct))"))
    @inbounds for (i,zψkd,) in enumerate(pdct)
        X[i] = f(FF, p.wp, sidx, θ, σ, zψkd..., itype...)
    end
end
