import Base: size

export dcdp_primitives, dcdp_Emax, dcdp_tmpvars, update_θ!, update_payoffs!, check_flowgrad, check_EVgrad, check_EVgrad!,
    _ψspace, _vspace

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

size(prim::dcdp_primitives) = _nz(prim), _nψ(prim), _nS(prim)

# --------------------- Emax --------------------------

struct dcdp_Emax{T<:Real,A1<:AbstractArray{T,3},A2<:AbstractArray{T,4},A3<:AbstractArray{T,4}}
    EV::A1
    dEV::A2
    dEV_σ::A3
end

function dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEV_σ::AbstractArray4{T}) where {T<:Real}
    dcdp_Emax{T,typeof(EV),typeof(dEV),typeof(dEV_σ)}(EV,dEV,dEV_σ)
end

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


# --------------------------- solve it! --------------------------------

function solve_vf_all!(EV::AbstractArray3, tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, roy::Real)
    update_payoffs!(tmp, θt, σv, p, roy, false)
    solve_vf_all!(EV, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)
end

function solve_vf_all!(EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4, tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, roy::Real, dograd::Bool=true)
    if dograd
        update_payoffs!(tmp, θt, σv, p, roy, true)
        solve_vf_all!(EV, dEV, dEV_σ, tmp.uin, tmp.uex, tmp.duin, tmp.duex, tmp.duexσ, tmp.ubVfull, tmp.dubVfull, tmp.dubV_σ, tmp.q, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, tmp.βdΠψ, p.β  )
    else
        solve_vf_all!(EV, tmp, θt, σv, p, roy)
    end
end

# wrappers
solve_vf_all!(evs::dcdp_Emax,                                                 tmp::dcdp_tmpvars, θt::AbstractVector, σv::Real, p::dcdp_primitives, roy::Real,                   dograd::Bool=true) = solve_vf_all!(evs.EV, evs.dEV, evs.dEV_σ, tmp, θt, σv, p, roy, dograd)
solve_vf_all!(evs::dcdp_Emax,                                                 tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, roy::Real, geoid::Integer=1, dograd::Bool=true) = solve_vf_all!(evs,                        tmp, _θt(θfull, geoid), _σv(θfull), p, roy, dograd)
solve_vf_all!(EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4, tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, roy::Real, geoid::Integer=1, dograd::Bool=true) = solve_vf_all!(EV, dEV, dEV_σ,             tmp, _θt(θfull, geoid), _σv(θfull), p, roy, dograd)
solve_vf_all!(EV::AbstractArray3,                                             tmp::dcdp_tmpvars, θfull::AbstractVector,        p::dcdp_primitives, roy::Real, geoid::Integer=1)                    = solve_vf_all!(EV,                         tmp, _θt(θfull, geoid), _σv(θfull), p, roy)

# ------------------------------ check total grad ------------------------------

reldiff(x::T, y::T) where {T<:Real} = x+y == zero(T) ? zero(T) : convert(T,2) * abs(x-y) / (abs(x)+abs(y))
absdiff(x::T, y::T) where {T<:Real} = abs(x-y)


function check_dEV(θt::AbstractVector{T}, σv::Real, prim::dcdp_primitives, roy::Real=0.2) where {T}
    evs = dcdp_Emax(θt, prim)
    tmp = dcdp_tmpvars(length(θt), prim)
    check_dEV!(evs, tmp, θt, σv, prim, roy)
end

function check_dEV!(evs::dcdp_Emax, tmp::dcdp_tmpvars, θt::AbstractVector{T}, σv::Real, prim::dcdp_primitives, roy::Real=0.2) where {T}
    check_size(θt, prim, evs)

    EV1 = zeros(T, size(evs.EV))
    EV2 = zeros(T, size(evs.EV))
    EVfd = similar(EV1)

    θ1, θ2 = similar(θt), similar(θt)
    nk = length(θt)

    solve_vf_all!(evs, tmp, θt, σv, prim, roy, true)

    for k in 1:nk
        θ1 .= θt
        θ2 .= θt
        h = max( abs(θt[k]), one(T) ) * cbrt(eps(T))
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]
        solve_vf_all!(EV1, tmp, θ1, σv, prim, roy)
        solve_vf_all!(EV2, tmp, θ2, σv, prim, roy)

        !all( evs.EV .== 0.0 )    ||  throw(error("EV all zeros"))
        !all( evs.dEV .== 0.0 )   ||  throw(error("dEV all zeros"))
        all(isfinite.(evs.dEV))   || throw(error("dEV not finite"))
        !all( EV1 .== 0.0 )   ||  throw(error("EV1 all zeros"))
        !all( EV2 .== 0.0 )   ||  throw(error("EV2 all zeros"))

        EVfd .= (EV2 .- EV1) ./ hh
        dEVk = @view(evs.dEV[:,:,k,:])
        EVfd ≈ dEVk  ||  warn("Bad grad in dim $k")

        absd = maximum( absdiff.(EVfd, dEVk ) )
        reld = maximum( reldiff.(EVfd, dEVk ) )
        println("In dimension $k, abs diff is $absd. max rel diff is $reld")
    end
end


function check_dEVσ(evs::dcdp_Emax, tmp::dcdp_tmpvars, θt::AbstractVector{T}, σv::T, p::dcdp_primitives, roy::Real) where {T}

    EV1 = zeros(T, size(evs.EV))
    EV2 = zeros(T, size(evs.EV))

    h = max( abs(σv), one(T) ) * cbrt(eps(T))
    σp = σv + h
    σm = σv - h
    hh = σp - σm

    update_payoffs!(tmp, θt, σv, p, roy, false; h=-h)
    solve_vf_all!(EV1, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)

    update_payoffs!(tmp, θt, σv, p, roy, false; h=h)
    solve_vf_all!(EV2, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)

    solve_vf_all!(evs, tmp, θt, σv, p, roy, true)

    dEVk = @view(evs.dEV_σ[:,:,1,1:end-1])
    EV1vw = @view(EV1[:,:,1:size(dEVk,3)])
    EV2vw = @view(EV2[:,:,1:size(dEVk,3)])
    EVfd = (EV2vw .- EV1vw) ./ hh
    EVfd ≈ dEVk  ||  warn("Bad grad for σ at vspace[1]")

    absd = maximum( absdiff.(EVfd, dEVk ) )
    reld = maximum( reldiff.(EVfd, dEVk ) )
    println("For σ, abs diff is $absd. max rel diff is $reld")
end


#
