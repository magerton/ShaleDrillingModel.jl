import Base: size

export dcdp_primitives,
    dcdp_Emax,
    dcdp_tmpvars,
    check_EVgrad,
    check_EVgrad!,
    _ψspace,
    _vspace


"""
make primitives. Note: flow payoffs, gradient, and grad wrt `σ` must have the following structure:
```julia
f(  θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, omroy::Real)
df( θ::AbstractVector{T}, σ::T,   z... , ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, omroy::Real)
dfσ(θ::AbstractVector{T}, σ::T,   z... , ψ::T, v::T,       d::Integer,                          omroy::Real)
```
"""
struct dcdp_primitives{T<:Real,AM<:AbstractMatrix{T},TT<:Tuple,AV<:AbstractVector{T}}
    f::Function
    df::Function
    dfσ::Function
    β::T
    wp::well_problem  # structure of endogenous choice vars
    zspace::TT        # z-space (tuple)
    Πz::AM            # transition for z
    # nψ::Int         # num ψ types (information)
    ψspace::AV        # ψspace = u + σv*v
    vspace::AV        # noise: ψ = u + σv*v
    ngeo::Int         # num geology types
end

# help us go from big parameter vector for all types to the relevant one
_σv(θ::AbstractVector) = θ[end]
_θt(θ::AbstractVector, geoid::Integer=1, ngeo::Integer=1) = vcat(θ[geoid], θ[ngeo+1:end-1])
_θt(θ::AbstractVector, geoid::Integer, prim::dcdp_primitives) = _θt(θ, geoid, prim.ngeo)

function _θt!(θt::AbstractVector, θfull::AbstractVector, geoid::Integer, ngeo::Integer=1)
    nfull = length(θfull)
    nt = length(θt)
    nt == nfull - ngeo  || throw(DimensionMismatch())
    1 <= geoid <= ngeo  || throw(DomainError())

    # updating
    θt[1] = θfull[geoid]
    @inbounds @simd for i = 2:nt
        θt[i] = θfull[i + ngeo]
    end
    return θfull[end]  # return σ
end

_nθt(θ::AbstractVector, ngeo::Integer) = length(θ)-ngeo
_nθt(θ::AbstractVector, prim::dcdp_primitives) = _nθt(θ, prim.ngeo)

_nz(prim::dcdp_primitives) = size(prim.Πz,1)
_nv(prim::dcdp_primitives) = length(prim.vspace)
_nψ(prim::dcdp_primitives) = length(prim.ψspace) # prim.nψ
_nS(prim::dcdp_primitives) = length(prim.wp)
_nSexp(prim::dcdp_primitives) = τmax(prim.wp)+1
_nd(prim::dcdp_primitives) = dmax(prim.wp)+1
_ndex(prim::dcdp_primitives) = exploratory_dmax(prim.wp)+1

# just in case we have things lying around...
_ψspace(prim::dcdp_primitives) = prim.ψspace
_ψspace(prim::dcdp_primitives, a, b) = _ψspace(prim)
_ψspace(prim::dcdp_primitives, a)    = _ψspace(prim)
_vspace(prim::dcdp_primitives) = prim.vspace
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


sprime_idx(prim::dcdp_primitives, i::Integer) = sprime_idx(prim.wp, i)
wp_info(prim::dcdp_primitives, i::Int) = wp_info(prim.wp, i)
state(prim::dcdp_primitives, i::Integer) = state(prim.wp, i)
dmax( prim::dcdp_primitives, i::Integer) = dmax(prim.wp, i)



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


size(prim::dcdp_primitives) = _nz(prim), _nψ(prim), _nS(prim)

# --------------------- Emax --------------------------

struct dcdp_Emax{T<:Real,A1<:AbstractArray{T,3},A2<:AbstractArray{T,4},A3<:AbstractArray{T,4}}
    EV::A1
    dEV::A2
    dEV_σ::A3
end

dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEV_σ::AbstractArray4{T}) where {T<:Real} =  dcdp_Emax{T,typeof(EV),typeof(dEV),typeof(dEV_σ)}(EV,dEV,dEV_σ)

function dcdp_Emax(θt::AbstractVector, p::dcdp_primitives{T}) where {T}
    EV   = zeros(T, _nz(p), _nψ(p),             _nS(p))
    dEV  = zeros(T, _nz(p), _nψ(p), length(θt), _nS(p))
    dEVσ = zeros(T, _nz(p), _nψ(p), _nv(p),     _nSexp(p)+1)
    dcdp_Emax(EV,dEV,dEVσ)
end

# --------------------- check conformable --------------------------

function check_size(θt::AbstractVector, prim::dcdp_primitives, evs::dcdp_Emax)
    nz, nψ, nS = size(prim)
    nθ, nv, nd = length(θt), _nv(prim), _nd(prim)
    ndex, nSexp = _ndex(prim), _nSexp(prim)

    prod(length.(prim.zspace)) == nz   ||  throw(error("Πz and zspace not compatible"))
    (nz,nψ,nS)         == size(evs.EV)   || throw(error("EV not conformable"))
    (nz,nψ,nθ,nS)      == size(evs.dEV)   || throw(error("dEV not conformable"))
    (nz,nψ,nv,nSexp+1) == size(evs.dEV_σ)   || throw(error("dEV_σ not conformable"))
    return true
end

# --------------------- tmp vars --------------------------

struct dcdp_tmpvars{T<:Real,AM<:AbstractMatrix{T}}
    uin::Array{T,4}
    uex::Array{T,3}

    duin::Array{T,5}
    duex::Array{T,4}
    duexσ::Array{T,4}

    ubVfull::Array{T,3}
    dubVfull::Array{T,4}
    dubV_σ::Array{T,4}

    q::Array{T,3}
    lse::Matrix{T}
    tmp::Matrix{T}
    βΠψ::Matrix{T}
    βdΠψ::Matrix{T}
    IminusTEVp::AM
end



function dcdp_tmpvars(nθt::Integer, prim::dcdp_primitives{T}) where {T}


    nz, nψ, nS = size(prim)
    nv, nd = _nv(prim), _nd(prim)
    ndex, nSexp = _ndex(prim), _nSexp(prim)

    nθt > nψ &&  throw(error("Must have more length(ψspace) > length(θt)"))

    # flow payoffs + gradients
    uin   = Array{T}(nz,nψ,nd,2)
    uex   = Array{T}(nz,nψ,nd)
    duin  = Array{T}(nz,nψ,nθt,nd,2)
    duex  = Array{T}(nz,nψ,nθt,ndex)
    duexσ = Array{T}(nz,nψ,nv,ndex)

    # choice-specific value functions
    ubVfull  = zeros(T,nz,nψ,nd)
    dubVfull = zeros(T,nz,nψ,nθt,nd)
    dubV_σ   = zeros(T,nz,nψ,nv,ndex)

    # other tempvars
    q        = Array{T}(nz,nψ,nd)
    lse      = Array{T}(nz,nψ)
    tmp      = Array{T}(nz,nψ)

    # transition matrices
    βΠψ = Matrix{T}(nψ,nψ)
    βdΠψ = similar(βΠψ)
    IminusTEVp = ensure_diagonal(prim.Πz)

    return dcdp_tmpvars(uin,uex,duin,duex,duexσ,ubVfull,dubVfull,dubV_σ,q,lse,tmp,βΠψ,βdΠψ,IminusTEVp)
end

dcdp_tmpvars(θfull::AbstractVector, prim::dcdp_primitives) = dcdp_tmpvars(_nθt(θfull,prim), prim)


#
