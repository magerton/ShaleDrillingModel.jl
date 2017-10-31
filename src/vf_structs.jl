import Base: size

export dcdp_primitives,
    dcdp_primitives_addlin,
    dcdp_primitives_add,
    dcdp_primitives_adddisc,
    dcdp_Emax,
    dcdp_tmpvars,
    check_EVgrad,
    check_EVgrad!,
    _zspace,
    _ψspace,
    _ψclamp,
    _nθt,
    _ngeo,
    _θt,
    _σv


"""
make primitives. Note: flow payoffs, gradient, and grad wrt `σ` must have the following structure:
```julia
f(  θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
df( θ::AbstractVector{T}, σ::T,   z... , ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
dfσ(θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
# dfψ(θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
```
"""
struct dcdp_primitives{T<:Real,AM<:AbstractMatrix{T},TT<:Tuple,AV<:AbstractVector{T}}
    f::Function
    dfθ::Function
    dfσ::Function
    dfψ::Function
    β::T
    wp::well_problem  # structure of endogenous choice vars
    zspace::TT        # z-space (tuple)
    Πz::AM            # transition for z
    # nψ::Int         # num ψ types (information)
    ψspace::AV        # ψspace = u + σv*v
    ngeo::Int         # num geology types
    nθt::Int          # Num parameters in flow payoffs MINUS 1 for σv
end


@inline dcdp_primitives_addlin( β::Real, wp::well_problem, zspace::NTuple{N,AbstractVector}, Πz::AbstractMatrix, ψspace::AbstractVector) where {N} = dcdp_primitives(u_addlin,  udθ_addlin,  udσ_addlin,  udψ_addlin,  β, wp, zspace, Πz, ψspace, 1,  6)
@inline dcdp_primitives_add(    β::Real, wp::well_problem, zspace::NTuple{N,AbstractVector}, Πz::AbstractMatrix, ψspace::AbstractVector) where {N} = dcdp_primitives(u_add,     udθ_add,     udσ_add,     udψ_add,     β, wp, zspace, Πz, ψspace, 10, 4)
@inline dcdp_primitives_adddisc(β::Real, wp::well_problem, zspace::NTuple{N,AbstractVector}, Πz::AbstractMatrix, ψspace::AbstractVector) where {N} = dcdp_primitives(u_adddisc, udθ_adddisc, udσ_adddisc, udψ_adddisc, β, wp, zspace, Πz, ψspace, 10, 5)


# help us go from big parameter vector for all types to the relevant one
_σv(θ::AbstractVector) = θ[end]

# This is b/c the type of SubArray differs depending on whether ngeo == geoid (linear indexing or not)
_θt(x::AbstractVector, nθt::Integer, ngeo::Integer, geoid::Integer, ::Type{Val{false}}, p1::Integer=0) = view(x, [geoid, ngeo+(1:nθt-1+p1)...])
_θt(x::AbstractVector, nθt::Integer, ngeo::Integer, geoid::Integer, ::Type{Val{true}} , p1::Integer=0) = view(x, ngeo+(0:nθt-1+p1))

_θt(x::AbstractVector, nθt::Integer, ngeo::Integer, geoid::Integer,         p1::Integer=0) = _θt(x,  nθt,        ngeo,       geoid, Val{ngeo        ∈ (1,geoid)}, p1)
_θt(x::AbstractVector, prim::dcdp_primitives, geoid::Integer,               p1::Integer=0) = _θt(x, _nθt(prim), _ngeo(prim), geoid, Val{_ngeo(prim) ∈ (1,geoid)}, p1)
_θt(x::AbstractVector, prim::dcdp_primitives, geoid::Integer, geogeo::Type, p1::Integer=0) = _θt(x, _nθt(prim), _ngeo(prim), geoid, geogeo,                       p1)


# functions to retrieve elements from dcdp_primitives
_nθt(   prim::dcdp_primitives) = prim.nθt
_ngeo(  prim::dcdp_primitives) = prim.ngeo
_nz(    prim::dcdp_primitives) = size(prim.Πz,1)
_nψ(    prim::dcdp_primitives) = length(prim.ψspace) # prim.nψ
_nS(    prim::dcdp_primitives) = _nS(prim.wp)
_nSexp( prim::dcdp_primitives) = _nSexp(prim.wp)
_nd(    prim::dcdp_primitives) = dmax(prim.wp)+1
_ndex(  prim::dcdp_primitives) = exploratory_dmax(prim.wp)+1

_zspace(prim::dcdp_primitives) = prim.zspace

# just in case we have things lying around...
_ψspace(prim::dcdp_primitives) = prim.ψspace
_ψspace(prim::dcdp_primitives, a, b) = _ψspace(prim)
_ψspace(prim::dcdp_primitives, a)    = _ψspace(prim)
_ψclamp(x::Real, prim::dcdp_primitives, a, b) = clamp(x, extrema(_ψspace(prim, a, b))...)
_ψclamp(x::Real, prim::dcdp_primitives, a) = clamp(x, extrema(_ψspace(prim, a))...)
_ψclamp(x::Real, prim::dcdp_primitives) = clamp(x, extrema(_ψspace(prim))...)

sprime_idx(prim::dcdp_primitives, i::Integer) = sprime_idx(prim.wp, i)
wp_info(prim::dcdp_primitives, i::Integer) = wp_info(prim.wp, i)
state(prim::dcdp_primitives, i::Integer) = state(prim.wp, i)
dmax( prim::dcdp_primitives, i::Integer) = dmax(prim.wp, i)


size(prim::dcdp_primitives) = _nz(prim), _nψ(prim), _nS(prim)

# --------------------- Emax --------------------------

struct dcdp_Emax{T<:Real,A1<:AbstractArray{T,3},A2<:AbstractArray{T,4}}
    EV::A1
    dEV::A2
    dEVσ::A1
    # dEV_ψ::A1
end

# dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEVσ::AbstractArray3{T}, dEV_ψ::AbstractArray3{T}) where {T} =  dcdp_Emax{T,typeof(EV),typeof(dEV)}(EV,dEV,dEVσ,dEV_ψ)
dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEVσ::AbstractArray3{T}) where {T} =  dcdp_Emax{T,typeof(EV),typeof(dEV)}(EV,dEV,dEVσ)

function dcdp_Emax(p::dcdp_primitives{T}) where {T}
    EV   = zeros(T, _nz(p), _nψ(p),          _nS(p))
    dEV  = zeros(T, _nz(p), _nψ(p), _nθt(p), _nS(p))
    dEVσ = zeros(T, _nz(p), _nψ(p),          _nSexp(p))
    # dEVψ = zeros(T, _nz(p), _nψ(p),          _nSexp(p))
    dcdp_Emax(EV,dEV,dEVσ) # ,dEVψ)
end

# --------------------- check conformable --------------------------

function check_size(prim::dcdp_primitives, evs::dcdp_Emax)
    nz, nψ, nS = size(prim)
    nθ, nd = _nθt(prim), _nd(prim)
    ndex, nSexp = _ndex(prim), _nSexp(prim)

    prod(length.(_zspace(prim))) == nz ||  throw(error("Πz and zspace not compatible"))
    (nz,nψ,nS)      == size(evs.EV)    ||  throw(error("EV not conformable"))
    (nz,nψ,nθ,nS)   == size(evs.dEV)   ||  throw(error("dEV not conformable"))
    (nz,nψ,nSexp)   == size(evs.dEVσ) ||  throw(error("dEVσ not conformable"))
    # (nz,nψ,nSexp)   == size(evs.dEV_ψ) ||  throw(error("dEV_ψ not conformable"))
    return true
end



# --------------------- tmp vars --------------------------

struct dcdp_tmpvars{T<:Float64,AM<:AbstractMatrix{Float64}}
    uin::Array{T,4}
    uex::Array{T,3}

    duin::Array{T,5}
    duex::Array{T,4}
    duexσ::Array{T,3}
    # duexψ::Array{T,3}

    ubVfull::Array{T,3}
    dubVfull::Array{T,4}
    dubV_σ::Array{T,3}
    # dubV_ψ::Array{T,3}

    q::Array{T,3}
    lse::Matrix{T}
    tmp::Matrix{T}
    Πψtmp::Matrix{T}
    IminusTEVp::AM
end

function check_size(prim::dcdp_primitives, t::dcdp_tmpvars)
    nz, nψ, nS = size(prim)
    nθ, nd = _nθt(prim), _nd(prim)
    ndex, nSexp = _ndex(prim), _nSexp(prim)

    # TODO: allow dmaxp1 to vary with regime

    (nz,nψ,nθ,nd) == size(t.duex)     || throw(DimensionMismatch())
    (nz,nψ,nd)    == size(t.q)        || throw(DimensionMismatch())
    (nz,nψ,nθ,nd) == size(t.dubVfull) || throw(DimensionMismatch())
    (nz,nψ,nd)    == size(t.dubV_σ)   || throw(DimensionMismatch())
    # (nz,nψ,nd)    == size(t.dubV_ψ)   || throw(DimensionMismatch())
end


function dcdp_tmpvars(prim::dcdp_primitives)

    T = Float64
    nz, nψ, nS = size(prim)
    nd = _nd(prim)
    nθt = _nθt(prim)
    ndex, nSexp = _ndex(prim), _nSexp(prim)

    nθt > nψ &&  throw(error("Must have more length(ψspace) > length(θt)"))

    # flow payoffs + gradients
    uin   = zeros(T,nz,nψ,nd,2)
    uex   = zeros(T,nz,nψ,nd)
    duin  = zeros(T,nz,nψ,nθt,nd,2)
    duex  = zeros(T,nz,nψ,nθt,ndex)
    duexσ = zeros(T,nz,nψ,ndex)
    # duexψ = zeros(T,nz,nψ,ndex)

    # choice-specific value functions
    ubVfull  = zeros(T,nz,nψ,nd)
    dubVfull = zeros(T,nz,nψ,nθt,nd)
    dubV_σ   = zeros(T,nz,nψ,ndex)
    # dubV_ψ   = zeros(T,nz,nψ,ndex)

    # other tempvars
    q        = zeros(T,nz,nψ,nd)
    lse      = zeros(T,nz,nψ)
    tmp      = zeros(T,nz,nψ)

    # transition matrices
    Πψtmp = Matrix{T}(nψ,nψ)
    IminusTEVp = ensure_diagonal(prim.Πz)

    # return dcdp_tmpvars(uin,uex,duin,duex,duexσ,duexψ,ubVfull,dubVfull,dubV_σ,dubV_ψ,q,lse,tmp,Πψtmp,IminusTEVp)
    return dcdp_tmpvars(uin,uex,duin,duex,duexσ,ubVfull,dubVfull,dubV_σ,q,lse,tmp,Πψtmp,IminusTEVp)
end


function zero!(t::dcdp_tmpvars)
    zero!(t.uin  )
    zero!(t.uex  )
    zero!(t.duin )
    zero!(t.duex )
    zero!(t.duexσ)
    # zero!(t.duexψ)
    zero!(t.ubVfull )
    zero!(t.dubVfull)
    zero!(t.dubV_σ  )
    # zero!(t.dubV_ψ  )
    zero!(t.q       )
    zero!(t.lse     )
    zero!(t.tmp     )
    zero!(t.Πψtmp   )
end

function zero!(evs::dcdp_Emax)
    zero!(evs.EV)
    zero!(evs.dEV)
    zero!(evs.dEVσ)
    # zero!(evs.dEV_ψ)
end


# ------------------------------ thoughts on if have adaptive grid for ψ -----------------------------


# _ψspace(prim::dcdp_primitives, minψ::Real, maxψ::Real) = linspace(minψ, maxψ, prim.nψ)
# _ψspace(prim::dcdp_primitives, ψextrema::NTuple{2}) = _ψspace(prim, ψextrema...)

# _vspace(           nsd::Real, n::Int) = linspace(-nsd, nsd, n)
# _ψspace(  σ::Real, nsd::Real, n::Int) = linspace(-nsd*(1.0+σ^2), nsd*(1.0+σ^2), n)
# _ψstep(   σ::Real, nsd::Real, n::Int) = 2.0 * nsd * (1.0+σ^2) / (n-1.0)
# _dψstepdσ(σ::Real, nsd::Real, n::Int) = 4.0 * nsd *      σ    / (n-1.0)
#
# _vspace(           prim::dcdp_primitives) = _vspace(     prim.maxsd, prim.nv)
# _ψspace(  σ::Real, prim::dcdp_primitives) = _ψspace(  σ, prim.maxsd, prim.nψ)
# _ψstep(   σ::Real, prim::dcdp_primitives) = _ψstep(   σ, prim.maxsd, prim.nψ)
# _dψstepdσ(σ::Real, prim::dcdp_primitives) = _dψstepdσ(σ, prim.maxsd, prim.nψ)


# midpoint(r::StepRangeLen) = (last(r) + first(r))/2.0
# function stepsrng(r::StepRangeLen)
#     m = midpoint(r)
#     s = step(r)
#     l = (first(r)-m)/s
#     return l : 1.0 : -l
# end
#
# function drngdlast(r::StepRangeLen)
#     l = last(r)
#     first(r) == -l  || throw(error("must be symmetric"))
#     return -1.0 : step(r)/l  : 1.0
# end
#
# function _dψdσ(ψspace::StepRangeLen, σ::Real)
#     l = last(ψspace)
#     first(ψspace) == -l  || throw(error("must be symmetric"))
#     twosig = 2.0 * σ
#     return -twosig : twosig * step(ψspace)/l  : twosig
# end

#
