using ShaleDrillingModel
using Test

wp1 = well_problem(4,4,5,3,2)
wp2 = well_problem(3,4,5,3,2)
nS = length(wp2)

wp2.endpts
wp2.SS[14:18]

@show idxd, idxs, horzn, s = ShaleDrillingModel.wp_info(wp2, 21)
@show ShaleDrillingModel._max_action(21, wp2)
(_dp1space(p,i), sprime_idx(p,i), horizon(p,i), state(p,i), )
