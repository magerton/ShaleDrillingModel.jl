export uflow, duflow, duflow_σ, fillflows, makepdct, check_flowgrad


function uflow(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,     d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0 && return zero(logp)
    u = exp(logp) * omroy * (Dgt0  ? θ[1] + ψ  :  θ[1] + ψ*ρ2_σ(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[4])
    return u
end


function duflow(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, d::Integer, k::Integer,     d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0  && return zero(T)
    k == 1  && return  d * exp(logp) * omroy
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T, d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end


function duflow_σ(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, v::T, d::Integer,     omroy::Real) where {T}
    d == 0  &&  return zero(T)
    ρ2 = ρ2_σ(σ)
    return d * exp(logp) * omroy * ρ2 * (v - 2.0*ρ2*σ*ψ)
end




function duflow!(du::AbstractVector{T}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, d::Integer,    d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    for k = 1:K
        du[k] = duflow(θ, σ,     logp, ψ, d, k,     d1, Dgt0, omroy)
    end
end


function duflow(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, d::Integer,    d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    duflow(Vector{T}(k), θ, σ,     logp, ψ, d,     d1, Dgt0, omroy)
    return du
end





# -----------------------------------------------------

function fillflows(Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector{T}, σ::T, f::Function, pdct::Base.Iterators.AbstractProdIterator, roy::T) where {T}
    omroy = one(T)-roy
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] = f(θ, σ, st..., 0, true , omroy)
        Xin1[i] = f(θ, σ, st..., 1, true , omroy)
        Xexp[i] = f(θ, σ, st..., 0, false, omroy)
    end
end

function fillflows(X::AbstractArray, θ::AbstractVector{T}, σ::T, f::Function, pdct::Base.Iterators.AbstractProdIterator, roy::T) where {T}
    omroy = one(T)-roy
    size(pdct) == size(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(θ, σ, st, omroy)
    end
end


function makepdct(zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, θ::AbstractVector, typ::Symbol)
    dspace = 1:dmax(wp)+1
    typ == :u   && return  Base.product(zspace..., ψspace,              dspace)
    typ == :du  && return  Base.product(zspace..., ψspace, 1:length(θ), dspace)
    typ == :duσ && return  Base.product(zspace..., ψspace, vspace,      dspace)
    throw(error("wrong type"))
end


# ------------------------ check flow grad --------------

function check_flowgrad(θ::AbstractVector{T}, σ::T, zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem) where {T}
    omroy = 1. - 0.2
    du = Vector{Float64}(length(θ))
    dufd = similar(du)
    updct = makepdct(zspace, ψspace, vspace, wp, θ, :u)
    for st in updct
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = uflow(θ, σ, st...,  d1D..., omroy)
            duflow!(du, θ, σ,  st...,  d1D..., omroy)
            Calculus.finite_difference!(u, θ, dufd, :central)
            dufd ≈ du || throw(error("Bad θ diff at $st, $d1D. du=$du and fd = $dufd"))
        end

        for v in vspace
            logp, ψ, d = st
            duσ = duflow_σ(θ, σ, logp, ψ, v, d, omroy)
            uσ(h::Real) = uflow(θ, σ+h, logp, ψ + h*v, d, 0, false, omroy)
            duσfd = Calculus.derivative(uσ, 0., :central)
            duσ ≈ duσfd  || throw(error("Bad σ diff at $st. duσ = $duσ and fd = $duσfd"))
        end

    end




end
