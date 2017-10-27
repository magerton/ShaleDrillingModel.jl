export duθ, duθ!, fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, θ::AbstractVector{T}, σ::T, df::Function,    st::Tuple,           d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = df(θ, σ, st[1:end-1]..., k, st[end], d1, Dgt0, itype...)
    end
end

function duθ(θ::AbstractVector{T}, σ::T,  df::Function,   st::Tuple,    d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    du!(Vector{T}(length(θ)), θ, σ, df, st, d1, Dgt0, itype...)
    return dx
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, h::Real) = k
tweak(ψ::Real   , h::Real) = ψ + h

function tweakstate(st::Tuple,  h::Real)
    h == zero(h) && return st
    return (st[1:end-2]..., tweak(st[end-1],h), st[end])
end

# fduσ(f::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, roy::Real, h::T) where {T} = u(θ, σ+h, st..., d1, Dgt0, roy)
fduψ(f::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, h::T, itype::Real...) where {T} = f(θ, σ, tweakstate(st, h)..., d1, Dgt0, itype...)

# ------------------------------------ wrapper for all flows   -----------------

makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:u}})  = Base.product(zspace..., ψspace,         0:dmax(wp))
makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:du}}) = Base.product(zspace..., ψspace, 1:nθt,  0:dmax(wp))

makepdct(zspace::Tuple, ψspace::StepRangeLen, wp::well_problem, θt::AbstractVector, typ::Type             ) = makepdct(zspace,      ψspace,        wp, length(θt), typ)
makepdct(p::dcdp_primitives,                                                        typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p,σ), p.wp, _nθt(p),    typ)
makepdct(p::dcdp_primitives,                                    nθt::Integer,       typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p,σ), p.wp, nθt,        typ)
makepdct(p::dcdp_primitives,                                    θt::AbstractVector, typ::Type, σ::Real=1.0) = makepdct(p, length(θt), typ, σ)

# ------------------------ check flow grad --------------

function check_flowgrad(θ::AbstractVector{T}, σ::T, f::Function, df::Function, dfσ::Function, dfψ::Function, zspace::Tuple, ψspace::Range, wp::well_problem, itype::Real...) where {T}
    K = length(θ)
    dx = Vector{T}(K)
    dxfd = similar(dx)

    isok = true

    for st in makepdct(zspace, ψspace, wp, θ, Val{:u})
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = f(θ, σ,     st..., d1D..., itype...)
            duθ!(dx, θ, σ, df, st,    d1D..., itype...)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            if !(dxfd ≈ dx)
                warn("Bad θ diff at $st, $d1D. du=$du and fd = $dufd")
                return false
            end
        end

        # check σ
        d1 = 0
        Dgt0 = false
        dσ = dfσ(θ, σ, st..., itype...)
        fdσ = Calculus.derivative((σh::Real) ->  f(θ, σh, st..., d1, Dgt0, itype...), σ, :central)
        if !(dσ ≈ fdσ)
            warn("Bad σ diff at $st. duσ = $dσ and fd = $fdσ")
            return false
        end

        # check ψ
        dψ = dfψ(θ, σ, st..., itype...)
        fdψ = Calculus.derivative((h::Real) -> fduψ(f, θ, σ, st, d1, Dgt0, h, itype...), 0.0, :central)
        if !(dψ ≈ fdψ)
            warn("Bad ψ diff at $st. duψ = $dψ and fdψ = $fdψ")
            return false
        end
    end
    return isok
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives, itype::Real...)  = check_flowgrad(θ, σ, p.f, p.dfθ, p.dfσ, p.dfψ, p.zspace, p.ψspace, p.wp, itype...)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives, itype::Real...) = check_flowgrad(_θt(θ, geoid), _σv(θ), p, itype...)

# ------------------------ fill flows --------------

function fillflows!(f::Function, X::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {T}
    length(pdct) == length(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(θ, σ, st..., itype...)
    end
end

function fillflows!(f::Function, Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {T}
    size(pdct) == size(Xin0) == size(Xin1) == size(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] = f(θ, σ, st..., 0, true , itype...)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xin1[i] = f(θ, σ, st..., 1, true , itype...)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xexp[i] = f(θ, σ, st..., 0, false, itype...)
    end
end

fillflows!(f::Function, uin::AbstractArray4, uex::AbstractArray3, θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) = @views fillflows!(f, uin[:,:,:,1], uin[:,:,:,2],   uex, θ, σ, pdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) =        fillflows!(p.f, t.uin,                    t.uex, θ, σ, pdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives,                   θ::AbstractVector, σ::Real,                                            itype::Real...) =        fillflows!(t, p, θ, σ, makepdct(p, θ, Val{:u}, σ),           itype...)

function fillflows_grad!(t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, itype::Real...)
    @views fillflows!(p.f,   t.uin[:,:,:,   1], t.uin[:,:,:,   2],  t.uex, θ, σ, makepdct(p, θ, Val{:u},  σ), itype...)
    @views fillflows!(p.dfθ, t.duin[:,:,:,:,1], t.duin[:,:,:,:,2], t.duex, θ, σ, makepdct(p, θ, Val{:du}, σ), itype...)
    fillflows!(p.dfσ, t.duexσ,                                             θ, σ, makepdct(p, θ, Val{:u},  σ), itype...)
    fillflows!(p.dfψ, t.duexψ,                                             θ, σ, makepdct(p, θ, Val{:u},  σ), itype...)
end




#
