using Pkg
# already done
# Pkg.add("Dynare") 
Pkg.add("AxisArrayTables")
Pkg.add("CSV")
Pkg.add("DataFrames")

using Dynare
using AxisArrayTables
using CSV
using DataFrames


# Запуск Dynare
context = @dynare "nk_model.mod"

output = context.results.model_results[1].simulations[1].data


# output = open("./nk_model/output/nk_model.jls", "r") do f
#     deserialize(f)
# end
println("!!!OUTPUT!!!")
println(output)
println(fieldnames(ModelResults))
println(fieldnames(Simulation))
println(typeof(output))
println(output[:pi])

df = DataFrame(output)

# Запись DataFrame в CSV
CSV.write("output.csv", df)