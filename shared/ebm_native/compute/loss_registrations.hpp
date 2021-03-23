// Steps for adding a new loss/objective function in C++:
//   1) Copy one of the existing Loss*.h include files into a new renamed Loss*.h file
//      (for regression, we recommend starting from LossRegressionPseudoHuber.h)
//   2) Modify the new Loss*.h file to handle the new loss function
//   3) Add [#include "Loss*.h"] to the list of other include files right below this guide
//   4) Add the Loss* type to the list of loss registrations in the RegisterLosses() function right below the includes
//   5) Modify the Register<Loss*>("loss_function_name", ...) entry to have the new loss function name
//      and the list of parameters needed for the loss function which are to be extracted from the loss string.
//   6) Update/verify that the constructor arguments on your Loss* class match the parameters in the loss registration
//      below. If the list of *LossParam items in the function RegisterLosses() do not match your constructor
//      parameters in the new Loss* struct, it will not compile and cryptic compile errors will be produced.
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

// Add new Loss*.h include files here:
#include "LossBinaryCrossEntropy.h"
#include "LossBinaryLogLoss.h"
#include "LossMulticlassCrossEntropy.h"
#include "LossMulticlassLogLoss.h"
#include "LossMultitaskBinaryLogLoss.h"
#include "LossMultitaskMulticlassCrossEntropy.h"
#include "LossMultitaskRegressionMse.h"
#include "LossRegressionMse.h"
#include "LossRegressionPseudoHuber.h"

// Add new Loss* type registrations to this list:
static INLINE_ALWAYS const std::vector<std::shared_ptr<const Registration>> RegisterLosses() {
   // IMPORTANT: the *LossParam types here must match the parameters types in your Loss* constructor
   return {
      RegisterLoss<LossMulticlassLogLoss>("log_loss"),
      RegisterLoss<LossRegressionPseudoHuber>("pseudo_huber", FloatParam("delta", 1))
      // TODO: add a "c_sample" here and adapt the instructions above to handle it
   };
}
