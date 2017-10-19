export du, du!, fillflows, makepdct, check_flowgrad, update_payoffs!

# -------------------------------- specific functions ----------------------------------------------

function u_add(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0 && return zero(logp)
    u = exp(logp) * omroy * (Dgt0  ? θ[1] + ψ  :  θ[1] + ψ*ρ2_σ(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[4])
    return u
end


function du_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  d * exp(logp) * omroy
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T, d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end


function duσ_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, v::T,                d::Integer, omroy::Real) where {T}
    d == 0  &&  return zero(T)
    ρ2 = ρ2_σ(σ)
    return d * exp(logp) * omroy * ρ2 * (v - 2.0*ρ2*σ*ψ)
end

# ----------------------------------- general wrappers -------------------------------------------

function du!(du::AbstractVector{T}, θ::AbstractVector{T}, σ::T, df::Function,    st::Tuple,           d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    for k = 1:K
        d = st[end]
        ψ = st[end-1]
        z = (st[1:end-2]...)
        du[k] = df(θ, σ, z..., ψ, k, d, d1, Dgt0, omroy)
    end
end

function du(θ::AbstractVector{T}, σ::T,  df::Function,   st::Tuple,    d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    dx = Vector{T}(length(θ))
    du!(dx, θ, σ, df, st, d1, Dgt0, omroy)
    return dx
end

# ------------------------------------ wrapper for all flows   -----------------

function fillflows(Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector{T}, σ::T, f::Function, pdct::Base.Iterators.AbstractProdIterator, roy::Real) where {T}
    omroy = one(T)-roy
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] = f(θ, σ, st..., 0, true , omroy)
        Xin1[i] = f(θ, σ, st..., 1, true , omroy)
        Xexp[i] = f(θ, σ, st..., 0, false, omroy)
    end
end

function fillflows(Xin0::AbstractArray3, Xin1::AbstractArray3, Xexp::AbstractArray3, θ::AbstractVector{T}, σ::T, f::Function, pdct::Base.Iterators.AbstractProdIterator, roy::Real, h::T, v::Real) where {T}
    σ += h
    omroy = one(T)-roy
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        z = st[1:end-2]
        ψ = st[end-1]
        d = st[end]
        Xin0[i] = f(θ, σ  , st...,             0, true , omroy)
        Xin1[i] = f(θ, σ  , st...,             1, true , omroy)
        Xexp[i] = f(θ, σ+h, z... , ψ + h*v, d, 0, false, omroy)
    end
end


function fillflows(X::AbstractArray, θ::AbstractVector{T}, σ::T, f::Function, pdct::Base.Iterators.AbstractProdIterator, roy::Real) where {T}
    omroy = one(T)-roy
    size(pdct) == size(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(θ, σ, st..., omroy)
    end
end


function makepdct(zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, θ::AbstractVector, typ::Symbol)
    dspace = 0:dmax(wp)
    typ == :u   && return  Base.product(zspace..., ψspace,              dspace)
    typ == :du  && return  Base.product(zspace..., ψspace, 1:length(θ), dspace)
    typ == :duσ && return  Base.product(zspace..., ψspace, vspace,      dspace)
    throw(error("wrong type"))
end



# ------------------------ check flow grad --------------

function check_flowgrad(θ::AbstractVector{T}, σ::T, f::Function, df::Function, dfσ::Function, zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, roy::Real) where {T}
    omroy = one(roy) - roy
    K = length(θ)
    dx = Vector{T}(K)
    dxfd = similar(dx)

    updct = makepdct(zspace, ψspace, vspace, wp, θ, :u)

    for st in updct
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = f(θ, σ,     st..., d1D..., omroy)
            du!(dx,  θ, σ, df, st,    d1D..., omroy)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            dxfd ≈ dx || throw(error("Bad θ diff at $st, $d1D. du=$du and fd = $dufd"))
        end

        for v in vspace
            z, ψ, d = (st[1:end-2]...), st[end-1], st[end]
            duσ = dfσ(θ, σ, z..., ψ, v, d, omroy)
            uσ(h::Real) = f(θ, σ+h, z..., ψ + h*v, d, 0, false, omroy)
            duσfd = Calculus.derivative(uσ, 0., :central)
            duσ ≈ duσfd  || throw(error("Bad σ diff at $st. duσ = $duσ and fd = $duσfd"))
        end

    end
end


# ------------------------ check flow grad --------------




function update_payoffs!(
    uin::AbstractArray4, uex::AbstractArray3, βΠψ::AbstractMatrix,
    f::Function,
    θt::AbstractVector, σv::Real, β::Real,
    roy::Real, zspace::Tuple, ψspace::AbstractVector, vspace::AbstractVector, wp::well_problem
    )

    uin0, uin1 = @view(uin[:,:,:,1]), @view(uin[:,:,:,2])
    fillflows(uin0, uin1, uex,    θt, σv, f,    makepdct(zspace, ψspace, vspace, wp, θt, :u),  roy)
    tauchen86_σ!(βΠψ, ψspace, σv)
    βΠψ .*= β
end


function update_payoffs!(
    uin::AbstractArray4, uex::AbstractArray3, βΠψ::AbstractMatrix,
    f::Function,
    θt::AbstractVector{T}, σv::Real, β::Real,
    roy::Real, zspace::Tuple, ψspace::AbstractVector, vspace::AbstractVector, wp::well_problem, h::T, v::T
    ) where {T}

    uin0, uin1 = @view(uin[:,:,:,1]), @view(uin[:,:,:,2])
    fillflows(uin0, uin1, uex,    θt, σv, f,    makepdct(zspace, ψspace, vspace, wp, θt, :u),  roy, h, v)
    tauchen86_σ!(βΠψ, ψspace, σv+h)
    βΠψ .*= β
end



function update_payoffs!(
    uin::AbstractArray4, uex::AbstractArray3, βΠψ::AbstractMatrix,
    duin::AbstractArray5, duex::AbstractArray4, duexσ::AbstractArray4, βdΠψ::AbstractMatrix,
    f::Function, df::Function, dfσ::Function,
    θt::AbstractVector, σv::Real, β::Real,
    roy::Real, zspace::Tuple, ψspace::AbstractVector, vspace::AbstractVector, wp::well_problem
    )

    uin0, uin1 = @view(uin[:,:,:,1]), @view(uin[:,:,:,2])
    duin0, duin1 = @view(duin[:,:,:,:,1]), @view(duin[:,:,:,:,2])
    fillflows(uin0, uin1, uex,    θt, σv, f,   makepdct(zspace, ψspace, vspace, wp, θt, :u), roy)
    fillflows(duin0, duin1, duex, θt, σv, df,  makepdct(zspace, ψspace, vspace, wp, θt, :du), roy)
    fillflows(duexσ,              θt, σv, dfσ, makepdct(zspace, ψspace, vspace, wp, θt, :duσ), roy)
    tauchen86_σ!(βΠψ, βdΠψ, ψspace, σv)
    βΠψ .*= β
    βdΠψ .*= β
end
