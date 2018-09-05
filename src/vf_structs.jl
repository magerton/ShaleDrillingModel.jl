import Base: size

export dcdp_primitives,
    dcdp_Emax,
    flow,
    dcdp_tmpvars,
    check_EVgrad,
    check_EVgrad!,
    _zspace,
    _ψspace,
    _ψ1clamp,
    _nθt,
    _θt,
    _σv

struct dcdp_primitives{FF,T<:Real,AM<:AbstractMatrix{T},TT<:Tuple,AV<:AbstractVector{T}}
    β::T
    wp::well_problem  # structure of endogenous choice vars
    zspace::TT        # z-space (tuple)
    Πz::AM            # transition for z
    ψspace::AV        # ψspace = u + σv*v
    nθt::Int          # Num parameters in flow payoffs MINUS 1 for σv
end

function dcdp_primitives(FF::Symbol, β::T, wp::well_problem, zspace::TT, Πz::AM, ψspace::AV) where {T,TT,AM,AV}
    FF ∈ (:exp,)    && return dcdp_primitives{Val{FF},T,AM,TT,AV}(β, wp, zspace, Πz, ψspace, 8)
    FF ∈ (:exproy,) && return dcdp_primitives{Val{FF},T,AM,TT,AV}(β, wp, zspace, Πz, ψspace, 9)
    # FF ∈ (:breaklin, :breakexp) && return dcdp_primitives{Val{FF},T,AM,TT,AV}(β, wp, zspace, Πz, ψspace, 13)
    throw(error("$FF is unknown"))
end

flow(prim::dcdp_primitives{FF}) where {FF} = FF

# help us go from big parameter vector for all types to the relevant one
_σv(θ::AbstractVector) = θ[end]

# allows for adding 1 unit for σv
_θt(x::AbstractVector, nθt::Integer,          p1::Integer=0) = view(x, 1:nθt+p1)
_θt(x::AbstractVector, prim::dcdp_primitives, p1::Integer=0) = _θt(x, _nθt(prim), p1)


# functions to retrieve elements from dcdp_primitives
_nθt(   prim::dcdp_primitives) = prim.nθt
_nz(    prim::dcdp_primitives) = size(prim.Πz,1)
_nψ(    prim::dcdp_primitives) = length(prim.ψspace) # prim.nψ
_nS(    prim::dcdp_primitives) = _nS(prim.wp)
_nSexp( prim::dcdp_primitives) = _nSexp(prim.wp)
_nd(    prim::dcdp_primitives) = dmax(prim.wp)+1
_ndex(  prim::dcdp_primitives) = exploratory_dmax(prim.wp)+1

_zspace(prim::dcdp_primitives) = prim.zspace

# just in case we have things lying around...
_ψspace(prim::dcdp_primitives) = prim.ψspace
@inline _ψ1clamp(u::Real,v::Real,ρ::Real, prim::dcdp_primitives) = clamp(_ψ1(u,v,ρ), extrema(_ψspace(prim))...)



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
end

dcdp_Emax(EV::AbstractArray3{T}, dEV::AbstractArray4{T}, dEVσ::AbstractArray3{T}) where {T} =  dcdp_Emax{T,typeof(EV),typeof(dEV)}(EV,dEV,dEVσ)

function dcdp_Emax(p::dcdp_primitives{FF,T}) where {FF,T}
    EV   = zeros(T, _nz(p), _nψ(p),          _nS(p))
    dEV  = zeros(T, _nz(p), _nψ(p), _nθt(p), _nS(p))
    dEVσ = zeros(T, _nz(p), _nψ(p),          _nSexp(p))
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
    Πψtmp = Matrix{T}(undef,nψ,nψ)
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


#
