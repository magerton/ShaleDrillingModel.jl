@testset "make State Space" begin

    dmx, Dmx, τ0mx, τ1mx, emx = (3, 4, 5, 3, 2)
    ep = ShaleDrillingModel.end_pts(    dmx, Dmx, τ0mx, τ1mx, emx)
    SS = ShaleDrillingModel.state_space(dmx, Dmx, τ0mx, τ1mx, emx)

    SS[ 1:ep[2] ]                                # Exploratory 1
    SS[ 1:ep[3] ]                                # Exploratory 0
    SS[  ep[3] .+ (1:dmx+1)]                     # Exploratory Terminal + learning update
    SS[  (τ0mx + τ1mx + dmx+3) .+ (1:2*Dmx-2)]   # Infill drilling
    SS[  (τ0mx + τ1mx + dmx+3)  +   2*Dmx-2+1]   # Terminal

    nS = length(SS)
    [ShaleDrillingModel._actionspace(i, dmx, Dmx, ep) for i in 1:nS]

    dDte = dmx,Dmx,ep
    a = SS, [
        (i,
        ShaleDrillingModel._regime(i,ep),
        ShaleDrillingModel._τ1(i,ep),
        ShaleDrillingModel._τ0(i,ep),
        ShaleDrillingModel._D(i,ep),
        ShaleDrillingModel._d1(i,ep),
        collect(ShaleDrillingModel._sprime(i,d,ep) for d in ShaleDrillingModel._actionspace(i,dDte...))
        ) for i in 1:nS]

    wp = well_problem(dmx, Dmx, τ0mx, τ1mx, emx)

    idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)..., ShaleDrillingModel.learn_state_inds(wp)...]

    @test idxs ⊆ 1:length(wp)
    @test 1:length(wp) ⊆ idxs

    for i in [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)...,]
        st = SS[i]
        i_of_st = state_idx(st.τ1, st.τ0, st.D, st.d1,     dmx, Dmx, τ0mx, τ1mx, emx)
        @test i_of_st == i
    end

end
