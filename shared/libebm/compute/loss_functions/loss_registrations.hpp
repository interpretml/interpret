// Copyright (c) 2023 The InterpretML Contributors
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifdef LOSS_REGISTRATIONS_HPP
#error loss_registrations.hpp is very special and should only be included once in a translation unit (*.cpp file).
#endif
#define LOSS_REGISTRATIONS_HPP

// Steps for adding a new loss/objective function in C++:
//   1) Copy one of the existing "*Loss.h" include files into a newly renamed "*Loss.h" file
//      (for regression, we recommend starting from PseudoHuberRegressionLoss.h)
//   2) Modify the new "*Loss.h" file to calculate the new loss function
//   3) Add [#include "*Loss.h"] to the list of other include files right below this guide
//   4) Add the new Loss type to the list of loss registrations in the RegisterLosses() function below
//   5) Modify the RegisterLoss<...>("loss_function_name", ...) entry to have the new loss function name
//      and the list of optional public parameters needed for the new Loss class.
//   6) Update/verify that the constructor arguments on your new Loss class match the parameters in the 
//      loss registration below. If the list of parameters in the function RegisterLosses() do not match your 
//      constructor parameters in your new Loss class, it will not compile and cryptic compile errors will be produced.
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit.

// Add new "*Loss.h" include files here:
#include "ExampleRegressionLoss.hpp"
#include "PseudoHuberRegressionLoss.hpp"
#include "RmseRegressionLoss.hpp"
#include "RmseLogLinkRegressionLoss.hpp"
#include "LogLossBinaryLoss.hpp"
#include "LogLossMulticlassLoss.hpp"
#include "PoissonRegressionLoss.hpp"
#include "GammaDevianceRegressionLoss.hpp"

// Add new *Loss type registrations to this list:
static const std::vector<std::shared_ptr<const Registration>> RegisterLosses() {
   // IMPORTANT: the parameter types listed here must match the parameters types in your Loss class constructor
   return {
      RegisterLoss<ExampleRegressionLoss>("example", FloatParam("param0", 0.0), FloatParam("param1", 1.0)),
      RegisterLoss<PseudoHuberRegressionLoss>("pseudo_huber", FloatParam("delta", 1.0)),
      RegisterLoss<RmseRegressionLoss>("rmse"),
      RegisterLoss<RmseLogLinkRegressionLoss>("rmse_log"),
      RegisterLoss<LogLossBinaryLoss>("log_loss"),
      RegisterLoss<LogLossMulticlassLoss>("log_loss"),
      RegisterLoss<PoissonDevianceRegressionLoss>("poisson_deviance", FloatParam("max_delta_step", 0.7)),
      RegisterLoss<GammaDevianceRegressionLoss>("gamma_deviance"),
   };
}
