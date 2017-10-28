dmx, Dmx, τmx = (5, 10, 7)
ep = ShaleDrillingModel.end_pts(dmx,Dmx,τmx)
SS = state_space(dmx,Dmx,τmx)

SS[1:τmx+1]                         # Exploratory
SS[  τmx+1+(1:dmx+1)]               # Exploratory Terminal + post exploratory
SS[  τmx+1+   dmx+1+(1:2*Dmx-2)]    # Infill drilling
SS[  τmx+1+   dmx+1+   2*Dmx-2+1]   # Terminal

nS = length(SS)
[ShaleDrillingModel._actionspace(i, dmx, Dmx, ep) for i in 1:nS]

dDte = dmx,Dmx,ep
a = SS, [
    (i,
    ShaleDrillingModel._regime(i,ep),
    ShaleDrillingModel._τ(i,ep),
    ShaleDrillingModel._D(i,ep),
    ShaleDrillingModel._d1(i,ep),
    collect(ShaleDrillingModel._sprime(i,d,ep) for d in ShaleDrillingModel._actionspace(i,dDte...))
    ) for i in 1:nS]



for (i,s) in enumerate(SS)
    for d in ShaleDrillingModel._actionspace(i,dmx,Dmx,ep)
        if ShaleDrillingModel._regime(i,ep) != :learn
            @test ShaleDrillingModel.OLDsprime(s,d,Dmx,dmx) == SS[ShaleDrillingModel._sprime(i,d,ep)]
        end
    end
end



wp = well_problem(3,6,10)

idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)..., ShaleDrillingModel.learn_state_inds(wp)...]

@test idxs ⊆ 1:length(wp)
@test 1:length(wp) ⊆ idxs





















#
