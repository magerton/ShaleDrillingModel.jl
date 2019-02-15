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
    _σv,
    print_summary

struct dcdp_primitives{FF,T<:Real,UP<:AbstractUnitProblem,AM<:AbstractMatrix{T},TT<:Tuple,AV<:AbstractVector{T}}
    β::T
    wp::UP            # structure of endogenous choice vars
    zspace::TT        # z-space (tuple)
    Πz::AM            # transition for z
    ψspace::AV        # ψspace = u + σv*v
    nθt::Int          # Num parameters in flow payoffs MINUS 1 for σv
    anticipate_e::Bool # do we anticipate the ϵ shocks assoc w/ each choice?
end

function dcdp_primitives(FF::Symbol, β::T, wp::UP, zspace::TT, Πz::AM, ψspace::AV, anticipate_e::Bool=true) where {T,TT,AM,AV,UP<:AbstractUnitProblem}
    dcdp_primitives{Val{FF},T,UP,AM,TT,AV}(β, wp, zspace, Πz, ψspace, number_of_model_parms(FF), anticipate_e)
end

flow(prim::dcdp_primitives{FF}) where {FF} = FF

# help us go from big parameter vector for all types to the relevant one
_σv(θ::AbstractVector) = last(θ)

# allows for adding 1 unit for σv
_θt(x::AbstractVector, nθt::Integer,          p1::Integer=0) = view(x, 1:nθt+p1)
_θt(x::AbstractVector, prim::dcdp_primitives, p1::Integer=0) = _θt(x, _nθt(prim), p1)

function print_summary(p::dcdp_primitives)
    print("\n--------------------------------------\n    Primitives\n--------------------------------------\n")
    println("Flow payoffs are $(flow(p)) with nθt+1 = $(_nθt(p))+1 parameters")
    println("Shocks anticipated? $(_anticipate_e(p))")
    println("β = $(p.β)")
    println("wp = $(p.wp)")
    println("zsize = $(length.(_zspace(p))) and zspace = $(_zspace(p))")
    println("nψ = $(_nψ(p)) and ψspace = $(_ψspace(p))")
    println("--------------------------------------")
end

# functions to retrieve elements from dcdp_primitives
_nθt(   prim::dcdp_primitives) = prim.nθt
_nz(    prim::dcdp_primitives) = size(prim.Πz,1)
_nψ(    prim::dcdp_primitives) = length(prim.ψspace) # prim.nψ
_nS(    prim::dcdp_primitives) = _nS(prim.wp)
_nSexp( prim::dcdp_primitives) = _nSexp(prim.wp)
_nd(    prim::dcdp_primitives) = _dmax(prim.wp)+1
_anticipate_e(prim::dcdp_primitives) = prim.anticipate_e

_zspace(prim::dcdp_primitives) = prim.zspace

# just in case we have things lying around...
_ψspace(prim::dcdp_primitives) = prim.ψspace
@inline _ψ1clamp(u::Real,v::Real,ρ::Real, prim::dcdp_primitives) = clamp(_ψ1(u,v,ρ), extrema(_ψspace(prim))...)



sprime_idx(prim::dcdp_primitives, i::Integer) = sprime_idx(prim.wp, i)
wp_info(prim::dcdp_primitives, i::Integer) = wp_info(prim.wp, i)
state(prim::dcdp_primitives, i::Integer) = state(prim.wp, i)
_dmax(prim::dcdp_primitives, i::Integer) = _dmax(prim.wp, i)
_dmax(prim::dcdp_primitives) = _dmax(prim.wp)

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
    nSexp = _nSexp(prim)

    prod(length.(_zspace(prim))) == nz ||  throw(error("Πz and zspace not compatible"))
    (nz,nψ,nS)      == size(evs.EV)    ||  throw(error("EV not conformable"))
    (nz,nψ,nθ,nS)   == size(evs.dEV)   ||  throw(error("dEV not conformable"))
    (nz,nψ,nSexp)   == size(evs.dEVσ) ||  throw(error("dEVσ not conformable"))
    # (nz,nψ,nSexp)   == size(evs.dEV_ψ) ||  throw(error("dEV_ψ not conformable"))
    return true
end



# --------------------- tmp vars --------------------------

struct dcdp_tmpvars{AA2<:AbstractMatrix{Float64}, AA3<:AbstractArray3{Float64}, AA4<:AbstractArray4{Float64}}
    ubVfull::AA3
    dubVfull::AA4
    dubV_σ::AA3

    q::AA3
    lse::Matrix{Float64}
    tmp::Matrix{Float64}
    tmp_cart::Matrix{CartesianIndex{3}}
    Πψtmp::Matrix{Float64}
    IminusTEVp::AA2
end

function check_size(prim::dcdp_primitives, t::dcdp_tmpvars)
    nz, nψ, nS = size(prim)
    nθ, nd = _nθt(prim), _nd(prim)
    nSexp =_nSexp(prim)

    # TODO: allow dmaxp1 to vary with regime

    (nz,nψ,nd)    == size(t.q)        || throw(DimensionMismatch())
    (nz,nψ,nθ,nd) == size(t.dubVfull) || throw(DimensionMismatch())
    (nz,nψ,nd)    == size(t.dubV_σ)   || throw(DimensionMismatch())
end


function dcdp_tmpvars(prim::dcdp_primitives)

    T = Float64
    nz, nψ, nS = size(prim)
    nd = _nd(prim)
    nθt = _nθt(prim)
    nSexp = _nSexp(prim)

    nθt > nψ &&  throw(error("Must have more length(ψspace) > length(θt)"))

    # choice-specific value functions
    ubVfull  = zeros(T,nz,nψ,nd)
    dubVfull = zeros(T,nz,nψ,nθt,nd)
    dubV_σ   = zeros(T,nz,nψ,nd)

    # other tempvars
    q        = zeros(T,nz,nψ,nd)
    lse      = zeros(T,nz,nψ)
    tmp      = zeros(T,nz,nψ)
    tmp_cart = Matrix{CartesianIndex{3}}(undef,nz,nψ)

    # transition matrices
    Πψtmp = Matrix{T}(undef,nψ,nψ)
    IminusTEVp = ensure_diagonal(prim.Πz)

    return dcdp_tmpvars(ubVfull, dubVfull, dubV_σ, q, lse, tmp, tmp_cart, Πψtmp, IminusTEVp)
end

function dcdp_tmpvars_view(t::dcdp_tmpvars, idxd::AbstractVector)
    first(idxd) == 1 || throw(DomainError())
    last(idxd) <= size(t.ubVfull,3) || throw(DomainError())

    @views ubV    = t.ubVfull[ :,:,  idxd]
    @views dubV   = t.dubVfull[:,:,:,idxd]
    @views dubV_σ = t.dubV_σ[  :,:,  idxd]
    @views q      = t.q[:,:,idxd]

    return dcdp_tmpvars(ubV, dubV, dubV_σ, q, t.lse, t.tmp, t.tmp_cart, t.Πψtmp, t.IminusTEVp)
end


function zero!(t::dcdp_tmpvars)
    zero!(t.ubVfull )
    zero!(t.dubVfull)
    zero!(t.dubV_σ  )
    zero!(t.q       )
    zero!(t.lse     )
    zero!(t.tmp     )
    zero!(t.Πψtmp   )
end

function zero!(evs::dcdp_Emax)
    zero!(evs.EV)
    zero!(evs.dEV)
    zero!(evs.dEVσ)
end


#
