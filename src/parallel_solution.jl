export parallel_solve_vf_all!, SharedEV


struct SharedEV{T<:Real,N,N2,TT<:Tuple}
    EV::SharedArray{T,N}
    dEV::SharedArray{T,N2}
    dEVσ::SharedArray{T,N2}
    itypes::TT
end


function SharedEV(pids::AbstractVector{<:Integer}, θ::AbstractVector{<:Real}, prim::dcdp_primitives{T}, itypes::AbstractVector...) where {T}

    nzdims = length(prim.zspace)
    zdims = length.(prim.zspace)
    ntypes = length(itypes)
    typedims = length.(itypes)

    N = nzdims + ntypes + 2

    nψ = _nψ(prim)
    nθ = _nθt(θ,prim)
    nv = _nv(prim)
    nS = _nS(prim)
    nSexp1 = _nSexp(prim)+1

    EV   = SharedArray{T}( (zdims..., nψ,     nS,     typedims...), init = S -> S[Base.localindexes(S)] = zero(T), pids=[1,pids...])
    dEV  = SharedArray{T}( (zdims..., nψ, nθ, nS,     typedims...), init = S -> S[Base.localindexes(S)] = zero(T), pids=[1,pids...])
    dEVσ = SharedArray{T}( (zdims..., nψ, nv, nSexp1, typedims...), init = S -> S[Base.localindexes(S)] = zero(T), pids=[1,pids...])

    return SharedEV{T,N,N+1,typeof(itypes)}(EV,dEV,dEVσ,itypes)
end

SharedEV(θ::AbstractVector, prim::dcdp_primitives, itypes::AbstractVector...) = SharedEV(addprocs(), θ, prim, itypes...)

"""
    dcdp_Emax(sev::SharedEV{T,N,N2}, typidx::Integer...) where {T,N,N2}

Return a `dcdp_Emax` object with (reshaped) slices of each array.
"""
function dcdp_Emax(sev::SharedEV{T,N,N2}, typidx::Integer...) where {T,N,N2}
    ntyps = length(typidx)
    ntyps == length(sev.itypes)  ||  throw(DimensionMismatch())

    dimz = N - ntyps - 2

    szEV = size(sev.EV)
    zdims = szEV[1:dimz]
    nψ, nS = szEV[dimz+1:dimz+2]
    typdims = length.(sev.itypes)

    nθ = size(sev.dEV, dimz+2)
    nv = size(sev.dEVσ, dimz+2)
    nSexp1 = size(sev.dEVσ, dimz+3)

    (zdims..., nψ,     nS,     typdims...) == size(sev.EV)   || throw(DimensionMismatch())
    (zdims..., nψ, nθ, nS,     typdims...) == size(sev.dEV)  || throw(DimensionMismatch())
    (zdims..., nψ, nv, nSexp1, typdims...) == size(sev.dEVσ) || throw(DimensionMismatch())

    colons(n) = ntuple((x)-> Colon(), n)
    nz = prod(zdims)

    @views EV   = reshape(sev.EV[  colons(dimz+2)..., typidx...], nz, nψ,     nS)
    @views dEV  = reshape(sev.dEV[ colons(dimz+3)..., typidx...], nz, nψ, nθ, nS)
    @views dEVσ = reshape(sev.dEVσ[colons(dimz+3)..., typidx...], nz, nψ, nv, nSexp1)

    typs = getindex.(sev.itypes, typidx)

    return dcdp_Emax(EV, dEV, dEVσ), typs
end

# -------------------------------------------------------------------

@GenGlobal g_SharedEV g_dcdp_primitives g_dcdp_Emax g_dcdp_tmpvars

function solve_vf_all!(sev::SharedEV, tmpv::dcdp_tmpvars, prim::dcdp_primitives, θ::AbstractVector, dograd::Type, typidx::Integer...; kwargs...)
    evs, typs = dcdp_Emax(sev, typidx...)
    solve_vf_all!(evs, tmpv, prim, θ, typs..., dograd; kwargs...)
end

# function solve_vf_all!(θ::AbstractVector, roy::Real, geoid::Integer, dograd::Type     ; kwargs...)
#     global g_dcdp_Emax
#     global g_dcdp_tmpvars
#     global g_dcdp_primitives
#     solve_vf_all!(g_dcdp_Emax, g_dcdp_tmpvars, g_dcdp_primitives, θ, roy, geoid, dograd; kwargs...)
# end

function solve_vf_all!(θ::AbstractVector, dograd::Type, typidx::Integer...; kwargs...)
    global g_SharedEV
    global g_dcdp_tmpvars
    global g_dcdp_primitives
    solve_vf_all!(g_SharedEV, g_dcdp_tmpvars, g_dcdp_primitives, θ, dograd, typidx...; kwargs...)
end


function parallel_solve_vf_all!(sev::SharedEV, θ::AbstractVector, dograd::Type; kwargs...)
    CR = CartesianRange( length.(sev.itypes) )
    s = @sync @parallel for Idx in collect(CR)
        solve_vf_all!(θ, dograd, Idx.I...; kwargs...)
    end
    return fetch(s)
end

# parallel_solve_vf_all!(sev::SharedEV, θ::AbstractVector, dograd::Bool) = parallel_solve_vf_all!(sev, θ, Val{dograd})



















#