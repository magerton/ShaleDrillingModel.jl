export duθ, duθ!, fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, FF::Type, θ::AbstractVector{T}, σ::T, st::Tuple,           d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = flowdθ(FF, θ, σ, st[1:end-1]..., k, st[end], d1, Dgt0, itype...)
    end
end

function duθ(FF::Type, θ::AbstractVector{T}, σ::T, st::Tuple,    d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    du!(Vector{T}(length(θ)), FF, θ, σ, st, d1, Dgt0, itype...)
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
flowfdψ(FF::Type, θ::AbstractVector{T},  σ::T, st::Tuple, d1::Integer, Dgt0::Bool, h::T, itype::Real...) where {T} = flow(FF, θ, σ, tweakstate(st, h)..., d1, Dgt0, itype...)

# ------------------------------------ wrapper for all flows   -----------------

makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:u}})  = Base.product(zspace..., ψspace,         0:dmax(wp))
makepdct(zspace::Tuple, ψspace::Range, wp::well_problem, nθt::Integer, ::Type{Val{:du}}) = Base.product(zspace..., ψspace, 1:nθt,  0:dmax(wp))

makepdct(zspace::Tuple, ψspace::StepRangeLen, wp::well_problem, θt::AbstractVector, typ::Type             ) = makepdct(zspace,      ψspace,      wp, length(θt), typ)
makepdct(p::dcdp_primitives,                                                        typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p), p.wp, _nθt(p),    typ)
makepdct(p::dcdp_primitives,                                    nθt::Integer,       typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p), p.wp, nθt,        typ)
makepdct(p::dcdp_primitives,                                    θt::AbstractVector, typ::Type, σ::Real=1.0) = makepdct(p, length(θt), typ, σ)

# ------------------------ check flow grad --------------

function check_flowgrad(FF::Type, θ::AbstractVector{T}, σ::T, zspace::Tuple, ψspace::Range, wp::well_problem, itype::Real...) where {T}
    K = length(θ)
    dx = Vector{T}(K)
    dxfd = similar(dx)

    isok = true

    for st in makepdct(zspace, ψspace, wp, θ, Val{:u})
        for d1D in [(0,true), (1,true), (0,false)]
            u(θ) = flow(FF, θ, σ, st..., d1D..., itype...)
            duθ!(dx,    FF, θ, σ, st,    d1D..., itype...)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            if !(dxfd ≈ dx)
                warn("FF = $FF. Bad θ diff at $st, $d1D. du=$dx and fd = $dxfd")
                return false
            end
        end

        # check σ
        d1 = 0
        Dgt0 = false
        dσ = flowdσ(FF, θ, σ, st..., itype...)
        fdσ = Calculus.derivative((σh::Real) ->  flow(FF, θ, σh, st..., d1, Dgt0, itype...), σ, :central)
        if !(dσ ≈ fdσ) && !isapprox(dσ,fdσ, atol= 1e-8)
            warn("Bad σ diff at $st. duσ = $dσ and fd = $fdσ")
            return false
        end

        # check ψ
        dψ = flowdψ(FF, θ, σ, st..., itype...)
        fdψ = Calculus.derivative((h::Real) -> flowfdψ(FF, θ, σ, st, d1, Dgt0, h, itype...), 0.0, :central)
        if !(dψ ≈ fdψ) && !isapprox(dψ, fdψ, atol=1e-7)
            warn("Bad ψ diff at $st. duψ = $dψ and fdψ = $fdψ")
            return false
        end
    end
    return isok
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, θ, σ, p.zspace, p.ψspace, p.wp, itype...)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, _θt(θ, p), _σv(θ), p, itype...)

# ------------------------ fill flows --------------

function fillflows!(FF::Type, f::Function, X::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {T}
    length(pdct) == length(X) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        X[i] = f(FF, θ, σ, st..., itype...)
    end
end


function fillflowrevs!(FF::Type, f::Function, Xin::AbstractArray, Xex::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {T}
    length(pdct) == length(Xin) == length(Xex) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin[i] = f(FF, θ, σ, st..., 0, true, itype...)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xex[i] = f(FF, θ, σ, st..., 1, false, itype...)
    end
end



function fillflows!(FF::Type, f::Function, Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector, σ::T, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {T}
    length(pdct) == length(Xin0) == length(Xin1) == length(Xexp) || throw(DimensionMismatch())
    @inbounds for (i, st) in enumerate(pdct)
        Xin0[i] = f(FF, θ, σ, st..., 0, true , itype...)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xin1[i] = f(FF, θ, σ, st..., 1, true , itype...)
    end
    @inbounds for (i, st) in enumerate(pdct)
        Xexp[i] = f(FF, θ, σ, st..., 0, false, itype...)
    end
end

# fill the flow-payoff (levels)
fillflows!(FF::Type, uin::AbstractArray4, uex::AbstractArray3, θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, itype::Real...)            = @views fillflows!(FF, flow, uin[:,:,:,1], uin[:,:,:,2],   uex, θ, σ, pdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives{FF},            θ::AbstractVector, σ::Real, pdct::Base.Iterators.AbstractProdIterator, itype::Real...) where {FF} =        fillflows!(FF, t.uin,                            t.uex, θ, σ, pdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives{FF},            θ::AbstractVector, σ::Real,                                            itype::Real...) where {FF} =        fillflows!(t, p, θ, σ,        makepdct(p, θ, Val{:u}, σ),           itype...)

function fillflows_grad!(t::dcdp_tmpvars, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, itype::Real...) where {FF}
    @views fillflows!(FF, flow,   t.uin[:,:,:,   1], t.uin[:,:,:,   2],  t.uex, θ, σ, makepdct(p, θ, Val{:u},  σ), itype...)
    @views fillflows!(FF, flowdθ, t.duin[:,:,:,:,1], t.duin[:,:,:,:,2], t.duex, θ, σ, makepdct(p, θ, Val{:du}, σ), itype...)
    fillflows!(       FF, flowdσ,                                      t.duexσ, θ, σ, makepdct(p, θ, Val{:u},  σ), itype...)
end




#
