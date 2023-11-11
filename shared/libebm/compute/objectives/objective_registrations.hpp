// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>


// Steps for adding a new objective in C++:
//   1) Copy one of the existing "*Objective.h" include files into a newly renamed "*Objective.h" file
//      (for regression, we recommend starting from ExampleRegressionObjective.h).
//   2) Change the name of the class and the constructor name to fit the new objective.
//   3) Update the parameters to the OBJECTIVE_BOILERPLATE macro for the new objective.
//   4) Modify the new "*Objective.h" file to calculate the new gradients, hessians, and metrics.
//   5) Add [#include "*Objective.h"] to the list of other include files right below this guide.
//   6) Add the new Objective type to the list of objective registrations in the RegisterObjectives() function below.
//   7) Modify the RegisterObjective<...>("objective_name", ...) entry to have the new objective name
//      and the list of optional public parameters needed for the new Objective class.
//   8) Update/verify that the constructor arguments on the new Objective class match the parameters in the 
//      objective registration below. If the list of parameters in the function RegisterObjectives() do not match the 
//      constructor parameters in the new Objective class, it will not compile and cryptic compile errors will be produced.
//   9) The bool value to the right of the class name in the template parameters for RegisterObjective determines
//      whether the objective can be executed with SIMD, or on a GPU. True means it can only operate with simple CPU
//      instructions and false means compile it for SIMD and GPU. The ExampleRegressionObjective is set by default
//      to only operate in CPU mode, but it is generally faster to set this to false if possible.
//   10) Recompile the C++ with either build.sh or build.bat depending on the operating system.
//   11) Enjoy your new Objective, and send us a PR on Github if you think others would benefit.

// Add new "*Objective.h" include files here:
#include "ExampleRegressionObjective.hpp"
#include "RmseRegressionObjective.hpp"
#include "RmseLogLinkRegressionObjective.hpp"
#include "PoissonDevianceRegressionObjective.hpp"
#include "TweedieDevianceRegressionObjective.hpp"
#include "GammaDevianceRegressionObjective.hpp"
#include "PseudoHuberRegressionObjective.hpp"
#include "LogLossBinaryObjective.hpp"
#include "LogLossMulticlassObjective.hpp"

// Add new *Objective type registrations to this list:
template<typename TFloat>
static const std::vector<std::shared_ptr<const Registration>> RegisterObjectives() {
   // IMPORTANT: the parameter types listed here must match the parameters types in the Objective class constructor
   return {
      Register<TFloat, ExampleRegressionObjective, ComputeFlags_Cpu>("example", FloatParam("param0", 0.0), FloatParam("param1", 1.0)),
      Register<TFloat, RmseRegressionObjective, ComputeFlags_ALL>("rmse"),
      Register<TFloat, RmseLogLinkRegressionObjective, ComputeFlags_ALL>("rmse_log"),
      Register<TFloat, PoissonDevianceRegressionObjective, ComputeFlags_ALL>("poisson_deviance"),
      Register<TFloat, TweedieDevianceRegressionObjective, ComputeFlags_ALL>("tweedie_deviance", FloatParam("variance_power", 1.5)),
      Register<TFloat, GammaDevianceRegressionObjective, ComputeFlags_ALL>("gamma_deviance"),
      Register<TFloat, PseudoHuberRegressionObjective, ComputeFlags_ALL>("pseudo_huber", FloatParam("delta", 1.0)),
      Register<TFloat, LogLossBinaryObjective, ComputeFlags_ALL>("log_loss"),
      Register<TFloat, LogLossMulticlassObjective, ComputeFlags_ALL>("log_loss"),
   };
}
