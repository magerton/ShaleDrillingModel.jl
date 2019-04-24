export starting_values

starting_values(x::AbstractPayoffFunction) = rand(length(x))
starting_values(x::StaticDrillingPayoff) = rand(length(x)+1)

starting_values(x::StaticDrillingPayoff{<:AbstractUnconstrainedDrillingRevenue, DrillingCost_constant, ExtensionCost_Constant}) = [-0x1.5abc9b2ad486fp+1, STARTING_log_ogip, STARTING_α_ψ, -0x1.efb4349a3e38cp+2,                        -0x1.6ce81fc7e1acep-3, 1.0, ]
starting_values(x::StaticDrillingPayoff{<:AbstractUnconstrainedDrillingRevenue, DrillingCost_dgt1,     ExtensionCost_Constant}) = [-0x1.30492070192dp+2,  STARTING_log_ogip, STARTING_α_ψ, -0x1.a8ca50a153ecbp+2, -0x1.45888b1d8fc67p+2, -0x1.c68737f1a1d98p-3, 1.0, ]

function starting_values(x::DrillingCost_TimeFE)
    if startstop(x.drillingcost) == (2008,2012,)
        timefe = [-13.1534, -9.18679, -7.90885, -7.16243, -6.77896,]
    else
        throw(error())
    end
    return vcat(timefe, 0x1.6a9e417fe7358p+0,)  # timeFE, multi-well discount
end

function starting_values(x::DrillingCost_TimeFE_rigrate)
    if startstop(x.drillingcost) == (2008,2012,)
        timefe = [-11.5378, -7.84324, -6.52969, -5.65049, -5.40226,]
    else
        throw(error())
    end
    return vcat(timefe, 1.47941, -0.811621,)  # timeFE, multi-well discount, rig-rate
end


function starting_values(x::StaticDrillingPayoff{<:AbstractUnconstrainedDrillingRevenue, C,  ExtensionCost_Constant}) where {C<:AbstractDrillingCost_TimeFE}
    return vcat(-0x1.a1c8146018faap+1, STARTING_log_ogip, STARTING_α_ψ, starting_values(x.drillingcost), 0x1.de94ce50afacep+0)
end

function starting_values(x::StaticDrillingPayoff{<:AbstractConstrainedDrillingRevenue})
    thet = starting_values(UnconstrainedProblem(x))
    deleteat!(thet, constrained_parms(x))
    return thet
end
