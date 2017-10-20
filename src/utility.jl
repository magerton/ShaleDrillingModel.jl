export du, du!, fillflows!, makepdct, check_flowgrad, update_payoffs!, fduσ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function du!(du::AbstractVector{T}, θ::AbstractVector{T}, σ::T, df::Function,    st::Tuple,           d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = df(θ, σ, st[1:end-2]..., st[end-1], k, st[end], d1, Dgt0, omroy)
    end
end

function du(θ::AbstractVector{T}, σ::T,  df::Function,   st::Tuple,    d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    du!(Vector{T}(length(θ)), θ, σ, df, st, d1, Dgt0, omroy)
    return dx
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, v::Real, h::Real) = k
tweak(ψ::Real   , v::Real, h::Real) = ψ + v*h

function tweakstate(st::Tuple, v::Real, h::Real)
    v == zero(v) ||  h == zero(h) && return st
    return (st[1:end-2]..., tweak(st[end-1],v,h), st[end])
end

fduσ(u::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, omroy::Real, v::Real, h::T) where {T} = u(θ, σ+h, tweakstate(st, v, h)..., d1, Dgt0, omroy)

# ------------------------------------ wrapper for all flows   -----------------

makepdct(zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:u}})   = Base.product(zspace..., ψspace,         0:dmax(wp))
makepdct(zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:du}})  = Base.product(zspace..., ψspace, 1:nθt,  0:dmax(wp))
makepdct(zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:duσ}}) = Base.product(zspace..., ψspace, vspace, 0:dmax(wp))

makepdct(zspace::Tuple, ψspace::StepRangeLen, vspace::Range, wp::well_problem, θt::AbstractVector, typ::Type             ) = makepdct(zspace,    ψspace,       vspace,      wp, length(θt), typ)
makepdct(p::dcdp_primitives,                                                   nθt::Integer,       typ::Type, σ::Real=1.0) = makepdct(p.zspace, _ψspace(p,σ), _vspace(p), p.wp, nθt,        typ)
makepdct(p::dcdp_primitives,                                                   θt::AbstractVector, typ::Type, σ::Real=1.0) = makepdct(p, length(θt), typ, σ)

# ------------------------ check flow grad --------------

function check_flowgrad(θ::AbstractVector{T}, σ::T, f::Function, df::Function, dfσ::Function, zspace::Tuple, ψspace::Range, vspace::Range, wp::well_problem, roy::Real) where {T}
    omroy = one(roy) - roy
    K = length(θ)
    dx = Vector{T}(K)
    dxfd = similar(dx)

    for st in makepdct(zspace, ψspace, vspace, wp, θ, Val{:u})
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = f(θ, σ,     st..., d1D..., omroy)
            du!(dx,  θ, σ, df, st,    d1D..., omroy)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            dxfd ≈ dx || throw(error("Bad θ diff at $st, $d1D. du=$du and fd = $dufd"))
        end
        for v in vspace
            duσ = dfσ(θ, σ, st[1:end-1]..., v, st[end]..., omroy)
            uσ(h::Real) = fduσ(f, θ, σ, st, 0, false, omroy, v, h)
            duσfd = Calculus.derivative(uσ, 0., :central)
            duσ ≈ duσfd  || throw(error("Bad σ diff at $st. duσ = $duσ and fd = $duσfd"))
        end
    end
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives, roy::Real)  = check_flowgrad(θ, σ, p.f, p.df, p.dfσ, p.zspace, p.ψspace, p.vspace, p.wp, roy)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives, roy::Real, geoid::Integer) = check_flowgrad(_θt(θ, geoid), _σv(θ), p, roy)

# ------------------------ fill flows --------------

function fillflows!(f::Function, X::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real) where {T}
    omroy = one(T)-roy
    size(pdct) == size(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(θ, σ, st..., omroy)
    end
end

function fillflows!(f::Function, Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real, v::Real=0.0, h::T=zero(T)) where {T}
    omroy = one(T)-roy
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] =     f( θ, σ, st..., 0, true , omroy)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xin1[i] =     f( θ, σ, st..., 1, true , omroy)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xexp[i] = fduσ(f, θ, σ, st   , 0, false, omroy, v, h)
    end
end

fillflows!(f::Function, uin::AbstractArray4, uex::AbstractArray3, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real, v::Real=0.0, h::T=0.0) where {T} = fillflows!(f, @view(uin[:,:,:,1]), @view(uin[:,:,:,2]), uex, θ, σ, pdct, roy, v, h)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real, v::Real=0.0, h::T=0.0) where {T} = fillflows!(p.f, t.uin,                                t.uex, θ, σ, pdct, roy, v, h)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::T,                                            roy::Real, v::Real=0.0, h::T=0.0) where {T} = fillflows!(t, p, θ, σ,             makepdct(p, θ, Val{:u}, σ),           roy, v, h)

function fillflows_grad!(t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, roy::Real)
    @views fillflows!(p.f,  t.uin[:,:,:,   1], t.uin[:,:,:,   2],  t.uex, θ, σ, makepdct(p, θ, Val{:u},  σ), roy)
    @views fillflows!(p.df, t.duin[:,:,:,:,1], t.duin[:,:,:,:,2], t.duex, θ, σ, makepdct(p, θ, Val{:du}, σ), roy)
    fillflows!(p.dfσ, t.duexσ,                                            θ, σ, makepdct(p, θ, Val{:duσ},σ), roy)
end




#
