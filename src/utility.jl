export fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, FF::AbstractPayoffFunction, wp::AbstractUnitProblem, θ::AbstractVector{T}, σ::T, z::Tuple, stinfo::Tuple, typestuff::Real...) where {T}
    K = length(θ)
    length(du) == K == length(FF) || throw(DimensionMismatch())
    stinfo[end]
    @inbounds for k = 1:K
        du[k] = flowdθ(FF, wp, θ, σ, z, stinfo[end-1], k, stinfo[end], typestuff...)
    end
end

# ------------------------ check flow grad --------------

function check_flowgrad(p::dcdp_primitives, θ::AbstractVector{T}, σ::T, itype::Real...) where {T}
    FF = p.f
    K = length(θ)
    K == length(FF) || throw(DimensionMismatch())
    dx = Vector{T}(undef, K)
    fdx = similar(dx)

    zspace = p.zspace
    ψspace = p.ψspace
    wp = p.wp

    zpdct = Base.product(zspace...)
    pdct = Base.product(zpdct, ψspace, 0:_dmax(wp))

    for zψd in pdct
        z, ψ, d = zψd

        for i ∈ 1:ShaleDrillingModel._nS(wp)
            u(θ) = flow(FF, θ, σ, wp, i, d, z, ψ, itype...)
            Calculus.finite_difference!(u, θ, fdx, :central)
            @inbounds for k ∈ 1:K
                dx[k] = flowdθ(FF, k, θ, σ, wp, i, d, z, ψ, itype...)
            end
            if !(fdx ≈ dx)
                @warn "FF = $FF. Bad θ diff at sidx=$sidx, (z,ψ,d)=$zψd. du=$dx and fd = $fdx"
                return false
            end
        end

        # check σ
        for sidx ∈ 1:min(2,end_ex0(wp))
            dσ = flowdσ(FF, θ, σ, wp, sidx, d, z, ψ, itype...)
            fdσ = Calculus.derivative((σh) -> flow(FF, θ, σh, wp, sidx, d, z, ψ, itype...), σ, :central)
            if !(dσ ≈ fdσ) && !isapprox(dσ, fdσ, atol= 1e-6)
                @warn "Bad σ diff at sidx=$sidx with (z,ψ,d)=$zψd. duσ = $dσ and fd = $fdσ"
                return false
            end
        end

        # check ψ
        for sidx in 1:min(4,end_ex0(wp))
            z,ψ,d = zψd
            dψ = flowdψ(FF, θ, σ, wp, sidx, d, z, ψ, itype...)
            fdψ = Calculus.derivative((h::Real) -> flow(FF, θ, σ, wp, sidx, d, z, ψ+h, itype...), 0.0, :central)
            if !(dψ ≈ fdψ) && !isapprox(dψ, fdψ, atol=1e-6)
                @warn "Bad ψ diff at sidx=$sidx with (z,ψ,d)=$(zψd). duψ = $dψ and fdψ = $fdψ"
                return false
            end
        end
    end
    return true
end

# ------------------------ fill flows --------------

function fillflows!(X::AbstractArray, f::Function, p::dcdp_primitives, θ::AbstractVector, σ::Real, sidx::Integer, itype::Real...)
    zpdct = Base.product(p.zspace...)
    idxd = actionspace(p.wp, sidx)
    pdct = Base.product(zpdct, p.ψspace, idxd)
    length(X) == length(pdct) || throw(DimensionMismatch("state[$sidx] is $(state(p.wp,sidx)). size(X) = $(size(X)) != size(pdct) = $(size(pdct))"))
    @inbounds for (i,(z,ψ,d,)) in enumerate(pdct)
        X[i] = f(p.f, θ, σ, p.wp, sidx, d, z, ψ, itype...)
    end
end

function fillflows_grad!(X::AbstractArray, f::Function, p::dcdp_primitives, θ::AbstractVector, σ::Real, sidx::Integer, itype::Real...)
    zpdct = Base.product(p.zspace...)
    idxd = 0:size(X, ndims(X))-1
    pdct = Base.product(zpdct, p.ψspace, Base.OneTo(length(θ)), idxd)
    length(X) == length(pdct) || throw(DimensionMismatch("state[$sidx] is $(state(p.wp,sidx)). size(X) = $(size(X)) != size(pdct) = $(size(pdct))"))
    @inbounds for (i,(z,ψ,k,d,),) in enumerate(pdct)
        X[i] = f(p.f, k, θ, σ, p.wp, sidx, d, z, ψ, itype...)
    end
end
