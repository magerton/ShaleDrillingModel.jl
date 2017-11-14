
function prDrill_infill!(
    EV::AbstractArray3, uin::AbstractArray4, uex::AbstractArray3, ubVfull::AbstractArray3,
    EU::AbstractArray3, rin::AbstractArray3, rex::AbstractArray3, ubUfull::AbstractArray3,
    P::AbstractArray4,

    lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,

    wp::well_problem, Πz::AbstractMatrix,

    Πψtmp::AbstractMatrix, ψspace::AbstractVector, σ::Real, β::Real
    )

    nz,nψ,nS = size(EV)
    dmaxp1 = dmax(wp)+1

    # ------------------------ size checks ----------------------------------

    (nz,nψ,nS) == size(EU)                       || throw(DimensionMismatch())
    (nz,nψ,dmaxp1,nS) == size(EU)                || throw(DimensionMismatch())
    (nz,nψ,dmaxp1,2) == size(uin)                || throw(DimensionMismatch())  # uin[z,ψ,d,d1]
    (nz,nz) == size(IminusTEVp) == size(Πz)      || throw(DimensionMismatch())
    (nz,nψ) == size(tmp) == size(lse)            || throw(DimensionMismatch())
    (nz,nψ,dmaxp1) == size(ubVfull) == size(rev) || throw(DimensionMismatch())


    # ------------------------ compute things ----------------------------------

    # terminal conditions
    EV[:,:,end] .= 0.0
    EU[:,:,end] .= 0.0

    # ----------- Infill drilling ----------------
    for i in ind_inf(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        @views EV0  = EV[:,:,i]
        @views EU0  = EU[:,:,i]
        @views ubV  = ubVfull[:,:,idxd]
        @views ubU  = ubUfull[:,:,idxd]
        @views ubV .= uin[:,:,idxd,1+s.d1] .+ β .* EV[:,:,idxs]
        @views ubU .= rin[:,:,idxd] .+ β .* EU[:,:,idxs]

        # Make probabilities
        softmax3!(ubV)
        P[:,:,1:dmaxp1,i] .= ubV

        # Form expected revenue = (I⊗Πz)∑_d q_d .* ( u .+ β .* EV)
        sumprod!(lse, ubV, ubU)
        A_mul_B_md!(EU0, lse, Πz, 2)

        # If infinite horizon, handle this
        if horzn == :Infinite
            for j in 1:size(EU0, ndims(EU0))
                @views update_IminusTVp!(IminusTEVp, Πz, β, ubV[:,j,1])
                fact = lufact(IminusTEVp)
                A_ldiv_B!(fact, EU0[:,j])
            end
        end
    end

    # ----------- Learning integration ----------------
    d2plus  = 2:dmax(wp)+1  # TODO: would be good to make this more general.
    lrn2inf = inf_fm_lrn(wp)
    exp2lrn = exploratory_learning(wp)
    ubU1    = @view(ubU[:,:,d2plus])
    EU1     = @view(EU[ :,:,lrn2inf])

    # ubVtilde = u[:,:,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]
    _βΠψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(ubU1, Πψtmp, EU1, 2)
    EU[:,:,exp2lrn] .= ubV1
    @views ubU1 .+= rex[:,:,d2plus]

    # ----------- Exploratory drilling ----------------
    for i in ind_exp(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)
        ip = action0(wp,i)

        @views EV0  = EV[:,:,i]
        @views EU0  = EU[:,:,i]

        @views ubV  = ubVfull[:,:,idxd]
        @views qvw  = q[:,:,idxd]
        @views ubU  = ubUfull[:,:,idxd]

        @views ubV[:,:,1] .= β .* EV[:,:,ip]
        @views ubU[:,:,1] .= β .* EU[:,:,ip]

        softmax3!(qvw, ubV)
        P[:,:,1:dmaxp1,i] .= qvw
        sumprod!(lse, qvw ubU)
        A_mul_B_md!(EU0, lse, Πz, 2)vvtkffjevvbkrvtecihjhrvirjhcfrkjdururjuuubhu
    end

end
