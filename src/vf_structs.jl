import Base: size

export dcdp_primitives, dcdp_Emax, dcdp_tmpvars, update_θ!, update_payoffs!, check_flowgrad, check_EVgrad

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
    nψ::Int           # num ψ types (information)
    vspace::AV        # noise: ψ = u + σv*v
    ngeo::Int         # num geology types
end

# help us go from big parameter vector for all types to the relevant one
_σv(θ::AbstractVector) = θ[end]
_θt(θ::AbstractVector, geoid::Integer=1, ngeo::Integer=1) = vcat(θ[geoid], θ[ngeo+1:end-1])
_θt(θ::AbstractVector, geoid::Integer, prim::dcdp_primitives) = _θt(θ, geoid, prim.ngeo)
_nθt(θ::AbstractVector, ngeo::Integer) = length(θ)-ngeo
_nθt(θ::AbstractVector, prim::dcdp_primitives) = _nθt(θ, prim.ngeo)

_nz(prim::dcdp_primitives) = size(prim.Πz,1)
_nv(prim::dcdp_primitives) = length(prim.vspace)
_nψ(prim::dcdp_primitives) = prim.nψ
_nS(prim::dcdp_primitives) = length(prim.wp)
_nSexp(prim::dcdp_primitives) = τmax(prim.wp)+1
_nd(prim::dcdp_primitives) = dmax(prim.wp)+1
_ndex(prim::dcdp_primitives) = exploratory_dmax(prim.wp)+1
_ψspace(prim::dcdp_primitives, minψ::Real, maxψ::Real) = linspace(minψ, maxψ, prim.nψ)
_ψspace(prim::dcdp_primitives, ψextrema::NTuple{2}) = _ψspace(prim, ψextrema...)


size(prim::dcdp_primitives) = _nz(prim), _nψ(prim), _nS(prim)

# --------------------- Emax --------------------------

struct dcdp_Emax{T<:Real,A1<:AbstractArray{T,3},A2<:AbstractArray{T,4},A3<:AbstractArray{T,4}}
    EV::A1
    dEV::A2
    dEV_σ::A3
end

dcdp_Emax(EV,dEV,dEV_σ) = dcdp_Emax{eltype(EV),typeof(EV),typeof(dEV),typeof(dEV_σ)}(EV,dEV,dEV_σ)

function dcdp_Emax(θt::AbstractVector, p::dcdp_primitives{T}) where {T}
    EV = Array{T}(  _nz(p), p.nψ, _nS(prim))
    dEV = Array{T}( _nz(p), p.nψ, length(θt), _nS(prim))
    dEVσ = Array{T}(_nz(p), p.nψ, _nv(prim), _nS(prim))
    dcdp_Emax(EV,dEV,dEV_σ)
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

    # flow payoffs + gradients
    uin   = Array{T}(nz,nψ,nd,2)
    uex   = Array{T}(nz,nψ,nd)
    duin  = Array{T}(nz,nψ,nθt,nd,2)
    duex  = Array{T}(nz,nψ,nθt,ndex)
    duexσ = Array{T}(nz,nψ,nv,ndex)

    # choice-specific value functions
    ubVfull  = Array{T}(nz,nψ,nd)
    dubVfull = Array{T}(nz,nψ,nθt,nd)
    dubV_σ   = Array{T}(nz,nψ,nv,ndex)

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


# --------------------------- payoff updating ---------------------------------

function update_payoffs!(tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, prim::dcdp_primitives, ψextrema::NTuple{2}, roy::Real=0.2, dograd::Bool=true)
    size(tmp.duin,3) == length(θt) || throw(DimensionMismatch())

    uin = tmp.uin
    uex = tmp.uex
    βΠψ = tmp.βΠψ

    uf = prim.f
    β = prim.β

    zspace = prim.zspace
    ψspace = _ψspace(prim, ψextrema)
    vspace = prim.vspace
    wp     = prim.wp

    if dograd
        duin  = tmp.duin
        duex  = tmp.duex
        duexσ = tmp.duexσ
        βdΠψ  = tmp.βdΠψ

        duf = prim.df
        dufσ = prim.dfσ
        update_payoffs!(uin, uex, βΠψ, duin, duex, duexσ, βdΠψ, uf, duf, dufσ, θt, σv, β, roy, zspace, ψspace, vspace, wp)
    else
        update_payoffs!(uin, uex, βΠψ,                          uf,            θt, σv, β, roy, zspace, ψspace, vspace, wp)
    end
end

update_payoffs!(tmp::dcdp_tmpvars, θfull::AbstractVector, prim::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, geoid::Integer, dograd::Bool) = update_payoffs!(tmp, _θt(θfull, geoid), _σv(θfull), prim, ψextrema, roy, dograd)

# --------------------------- check grad ---------------------------------

function check_flowgrad(θt::AbstractVector, σv::Real, p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real)
    ψspace = _ψspace(p, ψextrema)
    check_flowgrad(θt, σv, p.f, p.df, p.dfσ, p.zspace, ψspace, p.vspace, p.wp, roy)
end

check_flowgrad(θfull::AbstractVector, p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, geoid::Integer) = check_flowgrad(_θt(θfull, geoid), _σv(θfull), p, ψextrema, roy)


# --------------------------- solve it! --------------------------------

function solve_vf_all!(EV::AbstractArray3, tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real)
    update_payoffs!(tmp, θt, σv, p, ψextrema, roy, false)
    solve_vf_all!(EV, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)
end

function solve_vf_all!(EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4, tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, dograd::Bool=true)
    if dograd
        update_payoffs!(tmp, θt, σv, p, ψextrema, roy, true)
        solve_vf_all!(EV, dEV, dEV_σ, tmp.uin, tmp.uex, tmp.duin, tmp.duex, tmp.duexσ, tmp.ubVfull, tmp.dubVfull, tmp.dubV_σ, tmp.q, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, tmp.βdΠψ, p.β  )
    else
        solve_vf_all!(EV, tmp, θt, σv, p, ψextrema, roy)
    end
end

# wrappers
solve_vf_all!(evs::dcdp_Emax,                                                 tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real,                   dograd::Bool=true) = solve_vf_all!(evs.EV, evs.dEV, evs.dEV_σ, tmp, θt, σv, p, ψextrema, roy, dograd)
solve_vf_all!(evs::dcdp_Emax,                                                 tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, geoid::Integer=1, dograd::Bool=true) = solve_vf_all!(evs,                        tmp, _θt(θfull, geoid), _σv(θfull), p, ψextrema, roy, dograd)
solve_vf_all!(EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4, tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, geoid::Integer=1, dograd::Bool=true) = solve_vf_all!(EV, dEV, dEV_σ,             tmp, _θt(θfull, geoid), _σv(θfull), p, ψextrema, roy, dograd)
solve_vf_all!(EV::AbstractArray3,                                             tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, ψextrema::NTuple{2}, roy::Real, geoid::Integer=1)                    = solve_vf_all!(EV,                         tmp, _θt(θfull, geoid), _σv(θfull), p, ψextrema, roy)


# ------------------------------ check total grad ------------------------------




function check_EVgrad(θt::AbstractVector{T}, σv::Real, prim::dcdp_primitives, ψextrema::NTuple{2}, roy::Real=0.2) where {T}
    evs = dcdp_Emax(θt, prim)
    EV1 = similar(evs.EV)
    EV2 = similar(evs.EV)

    θ1, θ2 = similar(θt), similar(θt)
    nk = length(θt)

    tmp = dcdp_tmpvars(nk, prim)
    solve_vf_all!(evs, tmp, θt, σv, prim, ψextrema, roy)

    for k in 1:nk
        θ1 .= θt
        θ2 .= θt
        h = max( abs(θt[k]), one(T) ) * cbrt(eps(T))
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]
        solve_vf_all!(EV1, tmp, θ1, σv, prim, ψextrema, roy)
        solve_vf_all!(EV2, tmp, θ2, σv, prim, ψextrema, roy)

        all( (EV2 .- EV1) ./ hh .≈ @view(evs.dEV[:,:,k,:] ))   ||  throw(error("Bad grad"))
        !all( EV .== 0.0 )    ||  throw(error("All zeros"))
        !all( EV1 .== 0.0 )   ||  throw(error("All zeros"))
        !all( EV2 .== 0.0 )   ||  throw(error("All zeros"))
        extr = extrema( (EV2 .- EV1) ./ hh .- @view(evs.dEV[:,:,k,:] ) )
        println("In dimension $k, max absdiffs are $extr")
    end
end































#
