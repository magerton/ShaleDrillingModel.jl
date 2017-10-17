function solve_vf_terminal!(EV::AbstractArray{T}) where {T<:Real}
    EV[:,:,end] .= zero(T)
end

function solve_vf_terminal!(EV::AbstractArray{T}, dEV::AbstractArray{T}) where {T<:Real}
    solve_vf_terminal!(EV)
    length(dEV) > 0  &&  (dEV[:,:,:,end] .= zero(T))
end


function solve_vf_infill!(EV::AbstractArray, uin::AbstractArray, wp::well_problem, tmp1::AbstractMatrix, tmp2::AbstractMatrix, q::AbstractArray, ubVfull::AbstractArray, IminusTEVp::M, Πz::M, β::Real; maxit0::Int=12, maxit1::Int=20, vftol::Real=1e-11) where {M<:AbstractMatrix}

    nz,nψ,nS = size(EV)
    ndin, ndex = dmax(wp)+1, dmax(wp,1)+1

    (nz,nψ,ndin,2) == size(uin)                      || throw(DimensionMismatch())  # uin[z,ψ,d,d1]
    (nz,nψ,ndin)   == size(ubVfull)    == size(q)    || throw(DimensionMismatch())  # ubVfull[z,ψ,d]
    (nz,nψ)        == size(tmp1)       == size(tmp2) || throw(DimensionMismatch())
    (nz,nz)        == size(IminusTEVp) == size(Πz)   || throw(DimensionMismatch())

    dograd = false

    q0 = @view(q[:,:,1])

    for i in infill_state_inds(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        ubV  = @view( ubVfull[:,:,idxd])
        EV0  = @view( EV[:,:,i])
        ubV .= @view(uin[:,:,idxd,1+s.d1]) .+ β .* @view(EV[:,:,idxs])

        if dograd
            dubV = @view(dubVfull[:,:,:,idxd])
            dEV0 = @view(dEV[:,:,:,i])
            dubV .= @view(duin[:,:,:,idxd,1+s.d1]) .+ β .* @view(dEV[:,:,:,idxs])
        end

        if horzn == :Finite
            dograd  ||  vfit!(EV0, tmp1, tmp2, ubV, Πz)
            # dograd  &&  vfit!(EV0, dEV0, logsumubV, ubV, sumdubV, dubV, Πz)

        elseif horzn == :Infinite
            solve_inf_vfit!(EV0, tmp1, tmp2, ubV, Πz, β; maxit=maxit0, vftol=0.1)
            converged, iter, bnds = solve_inf_pfit!(EV0, tmp1, tmp2, q0, ubV, IminusTEVp, Πz, β; maxit=maxit1, vftol=1e-11)
            converged || throw(error("Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds"))
            # if dograd
            #     ubV[:,:,1] .= β .* EV0
            #     gradinf!(dEV0, ubV, sumdubV, Πz_sumdubV, dubV, IminusTEVp, Πz, β)
            # end
        else
            throw(error("horizon must be :Finite or :Infinite"))
        end
    end
end



function solve_vf_explore!(EV::AbstractArray, uex::AbstractArray, wp::well_problem, tmp1::AbstractArray, tmp2::AbstractArray, q::AbstractArray, ubVfull::AbstractArray, Πz::M, βΠψ::Matrix, β::Real) where {M<:AbstractMatrix}

    nz,nψ,nS = size(EV)
    ndin, ndex = dmax(wp)+1, dmax(wp,1)+1

    (nz,nψ,ndex) == size(uex)  == size(ubVfull) == size(q) || throw(DimensionMismatch())
    (nz,nψ)      == size(tmp1) == size(tmp2)               || throw(DimensionMismatch())
    (nz,nz)      == size(Πz)                               || throw(DimensionMismatch())
    (nψ,nψ)      == size(βΠψ)                              || throw(DimensionMismatch())

    dograd = false

    dmaxp1 = exploratory_dmax(wp)+1
    s_idx2_fm_1 = infill_state_idx_from_exploratory(wp)
    ubV  = @view( ubVfull[:,:,  1:dmaxp1])

    # update uβV
    A_mul_B_md!(@view(ubV[:,:,2:end]), βΠψ, @view(EV[:,:,s_idx2_fm_1]), 2)
    ubV[:,:,2:end] .+= @view(uex[:,:,2:end])
    if dograd
        dubV = @view(dubVfull[:,:,:,1:dmaxp1])
        A_mul_B_md!(@view(  duBv[:,:,:,2:end]),  βΠψ, @view(dEV[:,:,:,s_idx2_fm_1]), 2)
        A_mul_B_md!(@view(dubV_σ[:,:,:,2:end]), βdΠψ, @view( EV[:,:,  s_idx2_fm_1]), 2)
        dubV[  :,:,:,2:end] .+= @view(  duex[:,:,:,2:end])
        dubV_σ[:,:,:,2:end] .+= @view(duex_σ[:,:,:,2:end])
    end

    for i in explore_state_inds(wp)
        ip = action0(wp,i)
        ubV[:,:,1] .= @view(uex[:,:,1]) .+ β .* @view(EV[:,:,ip])
        if dograd
            dubV[:,:,:,1] .= @view(duex[:,:,:,1]) .+ β .* @view(dEV[:,:,:,ip])
            dubV_σ[:,:,:,1] .=  @view(dEV_σ[:,:,:,ip])
            vfit!(@view(EV[:,:,i]), @view(dEV[:,:,:,i]), logsumubV, ubV, q, sumduβV, dubV, Πz)
            sumprod(sumdubV_σ, duBV_σ, q)
            A_mul_B_md!(@view(dEV_σ[:,:,:,i]), Πz, sumdubV_σ, 1)
        else
            vfit!(@view(EV[:,:,i]), tmp1, tmp2, ubV, Πz)
        end
    end
end




# function vf_total_solve_learning(
#     EV::AbstractArray{<:Real,3},
#     uin::Array{<:Real,4}, uex::Array{<:Real,3},
#     dEV::AbstractArray, dEV_σ::AbstractArray,
#     duin::Array, duex::Array, duex_σ::Array,
#     Πz::AbstractMatrix,
#     βΠψ::AbstractMatrix,
#     βdΠψ::AbstractMatrix,
#     β::Real
#     )
#
#     s_idx1
#     s_idx2
#     s_idx2_fm_1
#
#     nZ,nψ,nS = size(EV)
#     ndin, ndex = dmax(wp), dmax(wp,1)
#
#     nz, nψ, ndin, 2 == size(uin)  || throw(DimensionMismatch())  # uin[z,ψ,d,d1]
#     nz, nψ, ndin == size(ubVfull) || throw(DimensionMismatch())  # ubVfull[z,ψ,d]
#     nz, nψ, ndin == size(q)       || throw(DimensionMismatch())
#     nz, nψ, ndex == size(uex)     || throw(DimensionMismatch())  # uex[z,ψ,d]
#
#     nz,nψ == size(logsumubV)           || throw(DimensionMismatch())
#     nz,nψ == size(EVtmp)               || throw(DimensionMismatch())
#     nz,nz == size(IminusTEVp)          || throw(DimensionMismatch())
#     nz,nz == size(Πz)                  || throw(DimensionMismatch())
#     nψ,nψ == size(βΠψ)                 || throw(DimensionMismatch())
#
#
#     dograd = length(dEV) > 0
#     if dograd
#         nθ, nv, nsex = size(dEV,3), size(dEV_σ,3), size(dEV_σ,4)
#
#         nz,nψ,nθ,ns == size(dEV)             || throw(DimensionMismatch()) #  dEV[z,ψ,θ,s]
#         nz,nψ,nv,nsex == size(dEV_σ)         || throw(DimensionMismatch()) #  dEV[z,ψ,v,s]
#         nz,nψ,nθ,ndin,2 == size(duin)        || throw(DimensionMismatch()) #  duin[z,ψ,θ,d,d1]
#         nz,nψ,nθ,ndex   == size(duex)        || throw(DimensionMismatch()) #  duex[z,ψ,θ,d]
#         nz,nψ,nv,ndex   == size(duex_σ)      || throw(DimensionMismatch()) #  duex_σ[z,ψ,v,d]
#         nψ, nψ          == size(βdΠψ)        || throw(DimensionMismatch())
#         nz,nψ,nθ,ndin == size(dubVfull)      || throw(DimensionMismatch())   # dubVfull
#         nz,nψ,nθ      == size(sumdubV)       || throw(DimensionMismatch())   # sumdubV[z,ψ,θ,1]
#         nz,nψ,nθ      == size(Πz_sumdubV)    || throw(DimensionMismatch())   # Πz_sumdubV
#         nz,nψ,nθ,ndex == size(dubV_σ)        || throw(DimensionMismatch())   # dubV_σ[z,ψ,v,idxd]
#         nz,nψ,nv      == size(sumdubV_σ)     || throw(DimensionMismatch())   # sumdubV_σ[z,ψ,v]
#     end
#
#     EV .= 0.0
#     dEV .= 0.0
#
#     for i,s in enumerate(s_idx2)
#         idxs = actions(wp,s)
#         idxd = Base.OneTo(length(idxs))
#
#         ubV  = @view( ubVfull[:,:,  idxd])
#         dubV = @view(dubVfull[:,:,:,idxd])
#         EV0  = @view( EV[:,:,  i])
#         dEV0 = @view(dEV[:,:,:,i])
#
#         ubV .= @view(uin[:,:,idxd,1+s.d1]) .+ β .* @view(EV[:,:,idxs])
#         dograd && dubV .= @view(duin[:,:,:,idxd,1+s.d1]) .+ β .* @view(dEV[:,:,:,idxs])
#
#         if horizon(wp,s) == :finite
#             dograd  ||  vfit!(EV0, logsumubV, ubV, Πz)
#             dograd  &&  vfit!(EV0, dEV0, logsumubV, ubV, sumdubV, dubV, Πz)
#
#         elseif horizon(wp,s) == :infinite
#             solve_inf_vfit!(EV0, EVtmp, logsumubV, ubv, Πz, β; maxit=10)
#             solve_inf_pfit!(EV0, EVtmp, logsumubV, ubv, @view(q[:,:,1]), IminusTEVp, Πz, β; maxit=20, vftol=1e-11)
#             if dograd
#                 ubV[:,:,1] .= β .* EV0
#                 gradinf!(dEV0, ubV, sumdubV, Πz_sumdubV, dubV, IminusTEVp, Πz, β)
#             end
#         end
#     end
#
#
#
#     idxd = exploratory_actions(wp)
#
#     ubV  = @view( ubVfull[:,:,  idxd])
#     dubV = @view(dubVfull[:,:,:,idxd])
#
#     # update uβV
#     A_mul_B_md!(@view(   ubV[:,:,  2:end]),  βΠψ, @view( EV[:,:,  s_idx2_fm_1]), 2)
#     A_mul_B_md!(@view(  duBv[:,:,:,2:end]),  βΠψ, @view(dEV[:,:,:,s_idx2_fm_1]), 2)
#     A_mul_B_md!(@view(dubV_σ[:,:,:,2:end]), βdΠψ, @view( EV[:,:,  s_idx2_fm_1]), 2)  # may have problems?
#     ubV[   :,:,  2:end] .+= @view(   uex[:,:,  2:end])
#     dubV[  :,:,:,2:end] .+= @view(  duex[:,:,:,2:end])
#     dubV_σ[:,:,:,2:end] .+= @view(duex_σ[:,:,:,2:end])
#
#     for i,s in enumerate(s_idx1)
#         ip = action0(wp,i)
#         ubV[:,:,1] .= @view(uex[:,:,1]) .+ β .* @view(EV[:,:,  ip])
#         if dograd
#             dubV[:,:,:,1] .= @view(duex[:,:,:,1]) .+ β .* @view(dEV[:,:,:,ip])
#             dubV_σ[:,:,:,1] .=  @view(dEV_σ[:,:,:,ip])
#             vfit!(@view(EV[:,:,i]), @view(dEV[:,:,:,i]), logsumubV, ubV, q, sumduβV, dubV, Πz)
#             sumprod(sumdubV_σ, duBV_σ, q)
#             A_mul_B_md!(@view(dEV_σ[:,:,:,i]), Πz, sumdubV_σ, 1)
#         else
#             vfit!(@view(EV[:,:,i]), logsumubV, ubV, q, Πz)
#         end
#     end
#
# end













#
