export duθ, duθ!, fillflows!, makepdct, check_flowgrad, update_payoffs!, fduψ, fillflows_grad!

# ----------------------------------- wrappers to fill a vector -------------------------------------------

function duθ!(du::AbstractVector{T}, FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, st::Tuple, d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    K = length(θ)
    length(du) == K  || throw(DimensionMismatch())
    @inbounds for k = 1:K
        du[k] = flowdθ(FF, θ, σ, z, st[end-1], k, st[end], d1, Dgt0, itype...)
    end
end

function duθ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, st::Tuple, d1::Integer, Dgt0::Bool, itype::Real...) where {T}
    dx = Vector{T}(length(θ))
    du!(dx, FF, θ, σ, z, st, d1, Dgt0, itype...)
    return dx
end

# ------------------------ tweak the state for finite differences --------------

tweak(k::Integer, h::Real) = k
tweak(ψ::Real   , h::Real) = ψ + h

function tweakstate(st::Tuple,  h::Real)
    h == zero(h) && return st
    return tweak(st[end-1],h), st[end]
end

# fduσ(f::Function, θ::AbstractVector{T}, σ::T, st::Tuple, d1::Integer, Dgt0::Bool, roy::Real, h::T) where {T} = u(θ, σ+h, st..., d1, Dgt0, roy)
flowfdψ(FF::Type, θ::AbstractVector{T},  σ::T, z::Tuple, st::Tuple, d1::Integer, Dgt0::Bool, sgnext::Bool, h::T, itype::Real...) where {T} = flow(FF, θ, σ, z, tweakstate(st, h)..., d1, Dgt0, sgnext, itype...)

# ------------------------------------ wrapper for all flows   -----------------

makepdct(zspace::Tuple, ψspace::AbstractRange, wp::well_problem, nθt::Integer, ::Type{Val{:u}})  = Base.product(zspace...), Base.product(ψspace,         0:dmax(wp))
makepdct(zspace::Tuple, ψspace::AbstractRange, wp::well_problem, nθt::Integer, ::Type{Val{:du}}) = Base.product(zspace...), Base.product(ψspace, 1:nθt,  0:dmax(wp))

makepdct(zspace::Tuple, ψspace::StepRangeLen, wp::well_problem, θt::AbstractVector, typ::Type             ) = makepdct(zspace,      ψspace,      wp, length(θt), typ)
makepdct(p::dcdp_primitives,                                                        typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p), p.wp, _nθt(p),    typ)
makepdct(p::dcdp_primitives,                                    nθt::Integer,       typ::Type, σ::Real=1.0) = makepdct(_zspace(p), _ψspace(p), p.wp, nθt,        typ)
makepdct(p::dcdp_primitives,                                    θt::AbstractVector, typ::Type, σ::Real=1.0) = makepdct(p, length(θt), typ, σ)

# ------------------------ check flow grad --------------

function check_flowgrad(FF::Type, θ::AbstractVector{T}, σ::T, zspace::Tuple, ψspace::AbstractRange, wp::well_problem, itype::Real...) where {T}
    K = length(θ)
    dx = Vector{T}(undef, K)
    dxfd = similar(dx)

    isok = true

    zpdct, stpdct = makepdct(zspace, ψspace, wp, θ, Val{:u})

    for st in stpdct, z in zpdct
        for d1Dgt0 in [(0,true,false,), (1,true,false,), (0,false,false,), (0,false,true,)]
            u(θ) = flow(FF, θ, σ, z, st..., d1Dgt0..., itype...)
            duθ!(dx,    FF, θ, σ, z, st,    d1Dgt0..., itype...)
            Calculus.finite_difference!(u, θ, dxfd, :central)
            if !(dxfd ≈ dx)
                @warn "FF = $FF. Bad θ diff at $st, $d1Dgt0. du=$dx and fd = $dxfd"
                return false
            end
        end

        # check σ
        d1 = 0
        Dgt0 = false
        sgnext = true
        dσ = flowdσ(FF, θ, σ, z, st..., itype...)
        fdσ = Calculus.derivative((σh::Real) ->  flow(FF, θ, σh, z, st..., d1, Dgt0, sgnext, itype...), σ, :central)
        if !(dσ ≈ fdσ) && !isapprox(dσ,fdσ, atol= 1e-8)
            @warn "Bad σ diff at $st. duσ = $dσ and fd = $fdσ"
            return false
        end

        # check ψ
        for sgnext in (true,false,)
            dψ = flowdψ(FF, θ, σ, z, st..., sgnext, itype...)
            fdψ = Calculus.derivative((h::Real) -> flowfdψ(FF, θ, σ, z, st, d1, Dgt0, sgnext, h, itype...), 0.0, :central)

            if !(dψ ≈ fdψ) && !isapprox(dψ, fdψ, atol=1e-7)
                @warn "Bad ψ diff at $st. duψ = $dψ and fdψ = $fdψ"
                return false
            end
        end
    end
    return isok
end

check_flowgrad(θ::AbstractVector, σ::Real, p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, θ, σ, p.zspace, p.ψspace, p.wp, itype...)
check_flowgrad(θ::AbstractVector,          p::dcdp_primitives{FF}, itype::Real...) where {FF} = check_flowgrad(FF, _θt(θ, p), _σv(θ), p, itype...)

# ------------------------ fill flows --------------

function fillflows!(FF::Type, f::Function, X::AbstractArray, θ::AbstractVector, σ::T, zpdct::Base.Iterators.ProductIterator, stpdct::Base.Iterators.ProductIterator, itype::Real...) where {T}
    length(zpdct) * length(stpdct) == length(X) || throw(DimensionMismatch())
    i = 0
    @inbounds for st in stpdct, z in zpdct
        X[i+=1] = f(FF, θ, σ, z, st..., itype...)
    end
end


function fillflowrevs!(FF::Type, f::Function, Xin::AbstractArray, Xex::AbstractArray, θ::AbstractVector, σ::T, zpdct::Base.Iterators.ProductIterator, stpdct::Base.Iterators.ProductIterator, itype::Real...) where {T}
    length(zpdct) * length(stpdct) == length(Xin) == length(Xex) || throw(DimensionMismatch())
    i = 0
    @inbounds for st in stpdct, z in zpdct
        Xin[i+=1] = f(FF, θ, σ, z, st..., 0, true, true, itype...)
    end
    i = 0
    @inbounds for st in stpdct, z in zpdct
        Xex[i+=1] = f(FF, θ, σ, z, st..., 1, false, true, itype...)
    end
end



function fillflows!(FF::Type, f::Function, Xin0::AbstractArray, Xin1::AbstractArray, Xexp::AbstractArray, θ::AbstractVector, σ::T, zpdct::Base.Iterators.ProductIterator, stpdct::Base.Iterators.ProductIterator, itype::Real...) where {T}
    length(zpdct) * length(stpdct)  == length(Xin0) == length(Xin1) == length(Xexp) || throw(DimensionMismatch())
    i = 0
    @inbounds for st in stpdct, z in zpdct
        Xin0[i+=1] = f(FF, θ, σ, z, st..., 0, true , false, itype...)
    end
    i = 0
    @inbounds for st in stpdct, z in zpdct
        Xin1[i+=1] = f(FF, θ, σ, z, st..., 1, true , false, itype...)
    end
    i = 0
    @inbounds for st in stpdct, z in zpdct
        Xexp[i+=1] = f(FF, θ, σ, z, st..., 0, false, true, itype...)
    end
end

# fill the flow-payoff (levels)
fillflows!(FF::Type, uin::AbstractArray4, uex::AbstractArray3, θ::AbstractVector, σ::Real, zpdct::Base.Iterators.ProductIterator, stpdct::Base.Iterators.ProductIterator, itype::Real...)            = @views fillflows!(FF, flow, uin[:,:,:,1], uin[:,:,:,2],   uex, θ, σ, zpdct, stpdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives{FF},            θ::AbstractVector, σ::Real, zpdct::Base.Iterators.ProductIterator, stpdct::Base.Iterators.ProductIterator, itype::Real...) where {FF} =        fillflows!(FF, t.uin,                            t.uex, θ, σ, zpdct, stpdct, itype...)
fillflows!(t::dcdp_tmpvars, p::dcdp_primitives{FF},            θ::AbstractVector, σ::Real,                                                                                itype::Real...) where {FF} =        fillflows!(t, p, θ, σ,        makepdct(p, θ, Val{:u}, σ)...,                 itype...)

function fillflows_grad!(t::dcdp_tmpvars, p::dcdp_primitives{FF}, θ::AbstractVector, σ::Real, itype::Real...) where {FF}
    @views fillflows!(FF, flow,   t.uin[:,:,:,   1], t.uin[:,:,:,   2],  t.uex, θ, σ, makepdct(p, θ, Val{:u},  σ)..., itype...)
    @views fillflows!(FF, flowdθ, t.duin[:,:,:,:,1], t.duin[:,:,:,:,2], t.duex, θ, σ, makepdct(p, θ, Val{:du}, σ)..., itype...)
    fillflows!(       FF, flowdσ,                                      t.duexσ, θ, σ, makepdct(p, θ, Val{:u},  σ)..., itype...)
end




#
