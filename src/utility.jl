export duθ, duθ!, fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, θ::AbstractVector{T}, σ::T, df::Function,    st::Tuple,           d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = df(θ, σ, st[1:end-1]..., k, st[end], d1, Dgt0, omroy)
    end
end

function duθ(θ::AbstractVector{T}, σ::T,  df::Function,   st::Tuple,    d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    du!(Vector{T}(length(θ)), θ, σ, df, st, d1, Dgt0, omroy)
    return dx
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, h::Real) = k
tweak(ψ::Real   , h::Real) = ψ + h

function tweakstate(st::Tuple,  h::Real)
    h == zero(h) && return st
    return (st[1:end-2]..., tweak(st[end-1],h), st[end])
end

# fduσ(f::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, omroy::Real, h::T) where {T} = u(θ, σ+h, st..., d1, Dgt0, omroy)
fduψ(f::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, omroy::Real, h::T) where {T} = f(θ, σ, tweakstate(st, h)..., d1, Dgt0, omroy)

# ------------------------------------ wrapper for all flows   -----------------

makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:u}})  = Base.product(zspace..., ψspace,         0:dmax(wp))
makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:du}}) = Base.product(zspace..., ψspace, 1:nθt,  0:dmax(wp))

makepdct(zspace::Tuple, ψspace::StepRangeLen, wp::well_problem, θt::AbstractVector, typ::Type             ) = makepdct(zspace,    ψspace,        wp, length(θt), typ)
makepdct(p::dcdp_primitives,                                    nθt::Integer,       typ::Type, σ::Real=1.0) = makepdct(p.zspace, _ψspace(p,σ), p.wp, nθt,        typ)
makepdct(p::dcdp_primitives,                                    θt::AbstractVector, typ::Type, σ::Real=1.0) = makepdct(p, length(θt), typ, σ)

# ------------------------ check flow grad --------------

function check_flowgrad(θ::AbstractVector{T}, σ::T, f::Function, df::Function, dfσ::Function, dfψ::Function, zspace::Tuple, ψspace::Range, wp::well_problem, roy::Real) where {T}
    omroy = one(roy) - roy
    K = length(θ)
    dx = Vector{T}(K)
    dxfd = similar(dx)

    isok = true

    for st in makepdct(zspace, ψspace, wp, θ, Val{:u})
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = f(θ, σ,     st..., d1D..., omroy)
            duθ!(dx, θ, σ, df, st,    d1D..., omroy)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            if !(dxfd ≈ dx)
                warn("Bad θ diff at $st, $d1D. du=$du and fd = $dufd")
                return false
            end
        end

        d1 = 0
        Dgt0 = false
        dσ = dfσ(θ, σ, st..., omroy)
        fdσ = Calculus.derivative((σh::Real) ->  f(θ, σh, st..., d1, Dgt0, omroy), σ, :central)
        if !(dσ ≈ fdσ)
            warn("Bad σ diff at $st. duσ = $dσ and fd = $fdσ")
            return false
        end

        dψ = dfψ(θ, σ, st..., omroy)
        fdψ = Calculus.derivative((h::Real) -> fduψ(f, θ, σ, st, d1, Dgt0, omroy, h), 0.0, :central)
        if !(dψ ≈ fdψ)
            warn("Bad ψ diff at $st. duψ = $dψ and fdψ = $fdψ")
            return false
        end
    end
    return isok
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives, roy::Real)  = check_flowgrad(θ, σ, p.f, p.dfθ, p.dfσ, p.dfψ, p.zspace, p.ψspace, p.wp, roy)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives, roy::Real, geoid::Integer) = check_flowgrad(_θt(θ, geoid), _σv(θ), p, roy)

# ------------------------ fill flows --------------

function fillflows!(f::Function, X::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real) where {T}
    omroy = one(T)-roy
    length(pdct) == length(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(θ, σ, st..., omroy)
    end
end

function fillflows!(f::Function, Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, roy::Real) where {T}
    omroy = one(T)-roy
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] = f(θ, σ, st..., 0, true , omroy)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xin1[i] = f(θ, σ, st..., 1, true , omroy)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xexp[i] = f(θ, σ, st..., 0, false, omroy)
    end
end

fillflows!(f::Function, uin::AbstractArray4, uex::AbstractArray3, θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, roy::Real) = @views fillflows!(f, uin[:,:,:,1], uin[:,:,:,2],   uex, θ, σ, pdct, roy)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, roy::Real) =        fillflows!(p.f, t.uin,                    t.uex, θ, σ, pdct, roy)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::Real,                                            roy::Real) =        fillflows!(t, p, θ, σ, makepdct(p, θ, Val{:u}, σ),           roy)

function fillflows_grad!(t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, roy::Real)
    @views fillflows!(p.f,   t.uin[:,:,:,   1], t.uin[:,:,:,   2],  t.uex, θ, σ, makepdct(p, θ, Val{:u},  σ), roy)
    @views fillflows!(p.dfθ, t.duin[:,:,:,:,1], t.duin[:,:,:,:,2], t.duex, θ, σ, makepdct(p, θ, Val{:du}, σ), roy)
    fillflows!(p.dfσ, t.duexσ,                                             θ, σ, makepdct(p, θ, Val{:u},  σ), roy)
    fillflows!(p.dfψ, t.duexψ,                                             θ, σ, makepdct(p, θ, Val{:u},  σ), roy)
end




#
