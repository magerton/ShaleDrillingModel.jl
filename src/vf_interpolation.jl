using Interpolations: Flag, Quadratic, InPlace

export ItpSharedEV, set_up_dcdp_workers

struct ItpSharedEV{T,A1<:Interpolations.AbstractInterpolation{T},A2<:Interpolations.AbstractInterpolation{T},A3<:Interpolations.AbstractInterpolation{T},TT<:Tuple}
  EV::A1
  dEV::A2
  dEVσ::A3
  itypes::TT
end

function ItpSharedEV(σ::Real, ψflag::Flag=Quadratic(InPlace()) )
  global g_SharedEV
  global g_dcdp_primitives
  return ItpSharedEV(g_SharedEV, g_dcdp_primitives, σ, ψflag)
end

@GenGlobal g_ItpSharedEV


function ItpSharedEV(sev::SharedEV{T,N,N2,TT}, p::dcdp_primitives, σ::Real=1.0; ψflag::Flag=Quadratic(InPlace())) where {T,N,N2,TT}

    # dimensions of things
    ntyp = length(sev.itypes)
    nθ = _nθt(p)
    nS = _nS(p)
    nSexp1 = _nSexp(p)

    # form spline specifications
    splinetype(r::Union{StepRange{<:AbstractFloat},StepRangeLen{<:AbstractFloat}}) = BSpline(Quadratic(InPlace()))
    splinetype(r::Union{LinRange{<:AbstractFloat}, UnitRange{<:Integer}}) = BSpline(Constant())
    splinetype(r::AbstractVector) = NoInterp()

    it_z     = splinetype.(p.zspace)
    it_itype = splinetype.(sev.itypes)

    it_EV   = (it_z..., BSpline(ψflag), NoInterp(),             it_itype...)
    it_dEV  = (it_z..., BSpline(ψflag), NoInterp(), NoInterp(), it_itype...)
    it_dEVσ = (it_z..., BSpline(ψflag), NoInterp(),             it_itype...)

    # form UNFILTERED spline objects.
    # These specify the SharedArray as the coef field and add type information about spline interpolation
    # if quadtratic, we will have to solve a system of linear equations later to interpolate
    itp_EV    = BSplineInterpolation(tweight(sev.EV  ), sev.EV  , it_EV  , OnCell(), Val{0}())
    itp_dEV   = BSplineInterpolation(tweight(sev.dEV ), sev.dEV , it_dEV , OnCell(), Val{0}())
    itp_dEVσ  = BSplineInterpolation(tweight(sev.dEVσ), sev.dEVσ, it_dEVσ, OnCell(), Val{0}())

    # information for how to scale the interpolation object
    scalegrid(x::AbstractRange{S}) where {S<:AbstractFloat} = StepRangeLen(x)
    scalegrid(x::UnitRange) = x
    scalegrid(x::Integer) = Base.OneTo(x)
    scalegrid(x::AbstractVector) = Base.OneTo(length(x))

    # scaled interpolation
    scl_EV   = scalegrid.((p.zspace..., _ψspace(p),         nS, sev.itypes...))
    scl_dEV  = scalegrid.((p.zspace..., _ψspace(p), nθ,     nS, sev.itypes...))
    scl_dEVσ = scalegrid.((p.zspace..., _ψspace(p),     nSexp1, sev.itypes...))

    sit_EV   = scale(itp_EV   , scl_EV...)
    sit_dEV  = scale(itp_dEV  , scl_dEV...)
    sit_dEVσ = scale(itp_dEVσ , scl_dEVσ...)
    # sit_dEVψ = Interpolations.scale(it_dEVψ , scl_dEVσ...)

    return ItpSharedEV{T,typeof(sit_EV),typeof(sit_dEV),typeof(sit_dEVσ),TT}(sit_EV, sit_dEV, sit_dEVσ, sev.itypes)
end


for typ in (:parallel, :serial)
    fun = Symbol("$(typ)_prefilterByView!")
    @eval begin
        function ($fun)(sev::SharedEV{T,N,N2}, isev::ItpSharedEV, dograd::Bool=true) where {T,N,N2}
            Ntyps = (length(sev.itypes[end]), )
            ($fun)(sev.EV, isev.EV.itp,         Ntyps...)
            if dograd
                ($fun)(sev.dEV, isev.dEV.itp,   Ntyps...)
                ($fun)(sev.dEVσ, isev.dEVσ.itp, Ntyps...)
            end
        end
    end
end

# function parallel_prefilterByView!(sev::SharedEV{T,N,N2}, isev::ItpSharedEV)
#     Ntyps = length.(sev.itypes)
#     parallel_prefilterByView!(sev.EV, isev.EV,     N- Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEV, isev.dEV,   N2-Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEVσ, isev.dEVσ, N- Ntyps+1:N...)
# end
#
# function parallel_prefilterByView!(sev::SharedEV{T,N,N2}, isev::ItpSharedEV)
#     Ntyps = length.(sev.itypes)
#     parallel_prefilterByView!(sev.EV, isev.EV,     N- Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEV, isev.dEV,   N2-Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEVσ, isev.dEVσ, N- Ntyps+1:N...)
# end

# TODO: MAKE SURE to check if data is within the indices

unsafegetityp(grid::AbstractArray, idx::Real) = idx
unsafegetityp(grid::Vector, idx::Integer) = grid[idx]

getitype(grid::Vector, idx::Integer) = unsafegetityp(grid, idx)

function getitype(grid::AbstractArray, idx::Real)
    minimum(grid) <= idx <= maximum(grid) || throw(DomainError(idx))
    unsafegetityp(grid, idx)
end


function unsafegetityp(sev::Union{SharedEV,ItpSharedEV}, itypidx::Tuple)
    length(sev.itypes) == length(itypidx) || throw(DimensionMismatch())
    return unsafegetityp.(sev.itypes, itypidx)
end












# -------------------------------------------------------------

function set_up_dcdp_workers(pids::AbstractVector{<:Integer}, prim::dcdp_primitives, typegrids::AbstractVector...; σ0::Real=1.0, ψflag::Interpolations.Flag=Quadratic(InPlace()), kwargs...)

    println("initialize shared arrays")
    sev = SharedEV(pids, prim, typegrids...)

    println("setting up tmpvars")
    @eval @everywhere begin
        # Inner loop setup
        set_g_dcdp_primitives($prim)
        set_g_dcdp_tmpvars( dcdp_tmpvars(get_g_dcdp_primitives()) )
        set_g_SharedEV($sev)

        set_g_ItpSharedEV( ItpSharedEV(get_g_SharedEV(), get_g_dcdp_primitives(), $σ0; ψflag=$ψflag) )
    end

    println("worker dcdp problems setup")

    # add a few checks
    fetch(@spawn size(get_g_SharedEV().EV))   == size(sev.EV)               ||  throw(error("remote sharedEV not created (?)"))
    fetch(@spawn get_g_SharedEV().EV  === get_g_ItpSharedEV().EV.itp.coefs) ||  throw(error("remote sharedEV not linked to ItpSharedEV"))
    fetch(@spawn get_g_SharedEV().EV) === sev.EV                            ||  throw(error("remote sharedEV not same object as local"))

    return sev, get_g_ItpSharedEV()
end
