
# ubV0  = @view( ubVfull[:,:,  1])
# dubV0 = @view( ubVfull[:,:,:,1])
# dubV_σ0 = @view(dubV_σ[:,:,:,1])
#
# ubV1  = @view( ubVfull[:,:,  idxd[2:end]])
# dubV1 = @view(dubVfull[:,:,:,idxd[2:end]])
# dubV_σ1 = @view(dubV_σ[:,:,:,2:end])



"""
    ensure_diagonal(Π::M)

If `M<:SparseMatrixCSC`, ensure that diagonal has all entries (even if zero).
"""
function ensure_diagonal(Π::M) where {T<:Number, M<:SparseMatrixCSC{T}}
    n = size(Π, 1)
    A = Π + speye(T,n)
    typeof(A) == M || throw(error("type of Πnew != typeof(Π)"))
    Arows = rowvals(A)
    Avals = nonzeros(A)
    for j = 1:n
       for i in nzrange(A, j)
          j == Arows[i]  &&  (Avals[i] -= one(T))
       end
    end
    return A
end

ensure_diagonal(Π::M) where {T<:Number, M<:Matrix{T}} = Π


function vfit!(EV0::AbstractArray, logsumubV::AbstractArray, ubVmax::AbstractArray, ubV::AbstractArray, Πz::AbstractMatrix)
    logsumexp3!(logsumubV,ubVmax,ubV)
    A_mul_B_md!(EV0,Πz,logsumubV,1)
end

function vfit!(EV0::AbstractArray, dEV0::AbstractArray, logsumubV::AbstractArray, ubVmax::AbstractArray, ubV::AbstractArray, sumdubV::AbstractMatrix, dubV::AbstractMatrix, Πz::AbstractMatrix)
    vfit!(EV0, dEV0, logsumubV, ubVmax, ubV, ubV, sumduβV, dubV, Πz)
end


function vfit!(EV0::AbstractArray, dEV0::AbstractArray, logsumubV::AbstractArray, ubVmax::AbstractArray, ubV::AbstractArray, q::AbstractArray, sumdubV::AbstractMatrix, dubV::AbstractMatrix, Πz::AbstractMatrix)
    logsumexp_and_softmax3!(logsumubV,q,ubVmax,ubV)
    A_mul_B_md!(EV0,Πz,logsumubV,1)
    sumdubV .= @view(dubV[:,:,1]) .* @view(q[:,:,1])
    for d in 2:size(dubV,ndims(dubV))
        sumdubV .+= @view(dubV[:,:,d]) .* @view(q[:,:,d])
    end
    A_mul_B_md!(dEV0,Πz,sumdubV,1)
end




function solve_inf_vfit!(EV0::AbstractArray, EVtmp::AbstractArray, logsumubV::AbstractArray, ubv::AbstractArray, Πz::AbstractMatrix, β::Real; maxit::Int=6, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        vfit!(EVtmp, logsumubV, EVtmp, ubV, Πz)
        bnds = extrema(EVtmp.-EV0) .* β ./ (1.0 .- β)
        ubV[:,:,1] .= β .* (EV0 .= EVtmp)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
    end
end


function pfit!(EV0::AbstractArray, EVtmp::AbstractArray, logsumubV::AbstractArray, q0::AbstractArray, ubV::AbstractArray, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real)
    logsumexp_and_softmax3!(logsumubV,q0,EVtmp,ubV)
    A_mul_B_md!(EVtmp, Πz, logsumubV, 1)
    EVminusEVtmp = logsumubV
    EVminusEVtmp .= EV0 .- EVtmp
    # FIXME: add convergence check
    # bnds = -extrema(EVminusEVtmp) .* -β ./ (1.0 .- β)
    # converged
    for j in 1:size(EV0, ndims(EV0))
        q0j = @view(q0[:,j])
        ΔEVj = @view(EVminusEVtmp[:,j])
        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lufact(IminusTEVp)
        A_ldiv_B!(fact, ΔEVj)                          # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= EVminusEVtmp                               # update V
    return extrema(EVminusEVtmp) .* -β ./ (1.0 .- β)   # get norm
end

function solve_inf_pfit!(EV0::AbstractArray, EVtmp::AbstractArray, logsumubV::AbstractArray, q0::AbstractArray, ubV::AbstractArray, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Int=20, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        bnds = pfit!(EV0, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
        ubV[:,:,1] .= β .* EV0
    end
end


function gradinf!(dEV0::AbstractArray, ubV::AbstractArray, ubVmax::AbstractArray, sumdubV::AbstractArray, Πz_sumdubV::AbstractArray, dubV::AbstractArray, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real)
    logsumexp_and_softmax3!(logsumubV,ubV,ubVmax,ubV)
    sum!(sumdubV, dubV)
    A_mul_B_md!(Πz_sumdubV, Πz, sumdubV, 1)
    for j in 1:size(sumubV, 2)
        q0j = @view(ubV[:,j,1])
        Πz_sumdubVj = @view(Πz_sumdubV[:,j,:])
        dEV0j = @view(dEV0[:,j,:])
        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lufact(IminusTEVp)
        A_ldiv_B!(dEV0j,fact,Πz_sumdubVj)
    end
end


function sumprod!(red::AbstractArray{T,3}, big::AbstractArray{T,4}, small::AbstractArray{T,3}) where {T}
    nz,nψ,nv,nd = size(big)
    (nz,nψ,nd,) == size(small) || throw(DimensionMismatch())
    (nz,nψ,nv,) == size(red)   || throw(DimensionMismatch())
    red .= zero(T)
    @inbounds for d in 1:nd, v in 1:nv, ψ in 1:nψ
        @simd for z in 1:nz
            red[z,ψ,v] += small[z,ψ,d] * big[z,ψ,v,d]
        end
    end
end
