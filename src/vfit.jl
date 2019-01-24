# --------------------------- basic VFIT ----------------------------

# simple Vfit
function vfit!(EV0::AbstractMatrix, ubV::AbstractArray, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix)
    logsumexp3!(lse,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
end

# preserves ubV & updates derivatives
function vfit!(EV0::AbstractMatrix, dEV0::AbstractArray3, ubV::AbstractArray3, dubV::AbstractArray4, q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix)
    logsumexp_and_softmax3!(lse,q,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
    sumdubV = @view(dubV[:,:,:,1])
    sumprod!(sumdubV,dubV,q)
    A_mul_B_md!(dEV0,Πz,sumdubV,1)
end

# destroys ubV and updates derivatives
function vfit!(EV0::AbstractMatrix, dEV0::AbstractArray3, ubV::AbstractArray3, dubV::AbstractArray4,                    lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix)
    vfit!(EV0, dEV0, ubV, dubV, ubV, lse, tmp, Πz)
end

# --------------------------- VFIT until conv ----------------------------

function solve_inf_vfit!(EV0::AbstractMatrix, ubV::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Integer=6, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        vfit!(tmp, ubV, lse, tmp, Πz)
        bnds = extrema(tmp.-EV0) .* β ./ (1.0 .- β)
        ubV[:,:,1] .= β .* (EV0 .= tmp)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
    end
end

# --------------------------- basic pfit ----------------------------

function pfit!(EV0::AbstractMatrix, ubV::AbstractArray3, ΔEV::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; vftol::Real=1e-11)
    # initial vfit
    q0 = @view(ubV[:,:,1])
    logsumexp_and_softmax3!(ΔEV,q0,tmp,ubV)
    A_mul_B_md!(tmp, Πz, ΔEV, 1)

    # compute difference & check bnds
    bnds = extrema(ΔEV .= EV0 .- tmp) .* -β ./ (1.0 .- β)
    if all(abs.(bnds) .< vftol)
        EV0 .= tmp
        return bnds
    end

    # full PFit
    for j in 1:size(EV0, 2)
        q0j = @view(q0[:,j])
        ΔEVj = @view(ΔEV[:,j])

        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lu(IminusTEVp)
        ldiv!(fact, ΔEVj)                          # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= ΔEV                               # update V
    return extrema(ΔEV) .* -β ./ (1.0 .- β)   # get norm
end

# --------------------------- pfit until convergence ----------------------------

function solve_inf_pfit!(EV0::AbstractMatrix, ubV::AbstractArray3, ΔEV::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Integer=30, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        bnds = pfit!(EV0, ubV, ΔEV, tmp, IminusTEVp, Πz, β; vftol=vftol)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
        ubV[:,:,1] .= β .* EV0
    end
end

# --------------------------- inf horizon gradient ----------------------------


# note -- this destroys ubV
function gradinf!(dEV0::AbstractArray3{T}, dEV0σ::AbstractMatrix{T}, ubV::AbstractArray3, dubV::AbstractArray4, dubV_σ::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real, do_σ::Bool) where {T}
    size(dubV,4) >= 2 || throw(DimensionMismatch("Need dubV with at least 2+ action possibilities"))
    nθ = size(dubV,3)

    @views sumdubV   = dubV[:,:,:,1]
    @views ΠsumdubV  = dubV[:,:,:,2]

    @views ΠsumdubVj = lse[:,1:nθ] # Array{T}(nz,nθ)
    @views dev0tmpj  = tmp[:,1:nθ] # Array{T}(nz,nθ)

    q = ubV

    softmax3!(q, lse, tmp )

    # for dubV/dθt
    sumprod!(sumdubV, dubV, q)
    A_mul_B_md!(ΠsumdubV, Πz, sumdubV, 1)

    # for dubV/dσ
    if do_σ
        @views sumdubV_σ   = dubV_σ[:,:,1]
        @views ΠsumdubV_σ  = dubV_σ[:,:,2]
        @views ΠsumdubV_σj = lse[:,1] # Array{T}(nz)
        @views dev0tmp_σj = tmp[:,1] # Array{T}(nz)

        sumprod!(sumdubV_σ, dubV_σ, q)
        A_mul_B_md!(ΠsumdubV_σ, Πz, sumdubV_σ, 1)
    end

    for j in 1:size(dubV,2)
        @views update_IminusTVp!(IminusTEVp, Πz, β, ubV[:,j,1])
        fact = lu(IminusTEVp)

        # Note: cannot do this with @view(dEV0[:,j,:])
        @views ΠsumdubVj .= ΠsumdubV[:,j,:]
        ldiv!(dev0tmpj, fact, ΠsumdubVj) # ΠsumdubV[:,j,:])
        dEV0[:,j,:] .= dev0tmpj

        if do_σ
            @views ΠsumdubV_σj .= ΠsumdubV_σ[:,j]
            ldiv!(dev0tmp_σj, fact, ΠsumdubV_σj) # ΠsumdubV[:,j,:])
            dEV0σ[:,j] .= dev0tmp_σj
        end
    end
end

function gradinf!(dEV0::AbstractArray3{T}, ubV::AbstractArray3, dubV::AbstractArray4, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real) where {T}
    gradinf!(dEV0, Array{T,2}(undef,0,0), ubV, dubV, Array{T,3}(undef,0,0,0), lse, tmp, IminusTEVp, Πz, β, false)
end

# --------------------------- double vfit/pfit loop -----------------------

function solve_inf_vfit_pfit!(EV0::AbstractMatrix, ubV::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; vftol::Real=1e-9, maxit0::Integer=40, maxit1::Integer=20)
    solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=maxit0, vftol=vftol)

    # try-catch loop in case we have insane parameters that have Pr(no action) = 0, producing a singular IminusTEVp matrix.
    converged, iter, bnds = try
        solve_inf_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=maxit1, vftol=vftol)
    catch
        @warn "pfit threw error. trying vfit."
        solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=5000, vftol=vftol)
    end

    converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds"
end

# --------------------------- helper function  ----------------------------


function sumprod!(red::AbstractArray3{T}, big::AbstractArray4, small::AbstractArray3) where {T}
    nz,nψ,nv,nd = size(big)
    (nz,nψ,nd,) == size(small) || throw(DimensionMismatch())
    (nz,nψ,nv,) == size(red)   || throw(DimensionMismatch())

    # can't do this b/c of src/vfit.jl#13-14 above where red = big[:,:,:,1]
    # fill!(red, zero(T))
    # @inbounds for d in 1:nd, v in 1:nv
    #     @views red[:,:,v] .+= small[:,:,d] .* big[:,:,v,d]
    # end

    # first loop w/ equals
    @inbounds for v in 1:nv
        @views red[:,:,v] .= small[:,:,1] .* big[:,:,v,1]
    end

    # second set w/ plus equals
    @inbounds for d in 2:nd, v in 1:nv
        @views red[:,:,v] .+= small[:,:,d] .* big[:,:,v,d]
    end
end


function sumprod!(red::AbstractMatrix{T}, big::AbstractArray3, small::AbstractArray3) where {T}
    nz,nψ,nd = size(big)
    (nz,nψ,nd) == size(small) || throw(DimensionMismatch())
    (nz,nψ) == size(red) || throw(DimensionMismatch())

    fill!(red, zero(T))
    @inbounds for d in 1:nd
        @views red .+= small[:,:,d] .* big[:,:,d]
    end
end
