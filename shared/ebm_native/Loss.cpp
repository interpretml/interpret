// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stddef.h> // size_t, ptrdiff_t
#include <vector>
#include <functional>
#include <memory>
#include <algorithm>

#include "EbmInternal.h" // INLINE_ALWAYS
#include "Logging.h" // EBM_ASSERT & LOG
#include "FeatureGroup.h"
#include "ThreadStateBoosting.h"

#include "Loss.h"
#include "RegisterLoss.h"

// Steps for adding a new loss/objective function in C++:
//   1) Copy one of the existing Loss*.h include files into a new renamed Loss*.h file
//   2) Modify the new Loss*.h file to handle your new Loss function
//   3) Add [#include "Loss*.h"] to the list of other include files right below this
//   4) Add the Loss* type to the list of loss registrations in RegisterLosses() right below this
//      IMPORTANT: the list of *LossParam items in the function RegisterLosses() below MUST match the constructor 
//                 parameters in the Loss* struct, otherwise it will not compile!
//   5) Recompile the C++ with either build.sh or build.bat depending on your operating system
//   6) Enjoy your new Loss function, and send us a PR on Github if you think others would benefit  :-)

// Add any new Loss*.h include files here:
#include "LossBinaryCrossEntropy.h"
#include "LossBinaryLogLoss.h"
#include "LossMulticlassCrossEntropy.h"
#include "LossMulticlassLogLoss.h"
#include "LossMultitaskBinaryLogLoss.h"
#include "LossMultitaskMulticlassCrossEntropy.h"
#include "LossMultitaskRegressionMse.h"
#include "LossRegressionMse.h"
#include "LossRegressionPseudoHuber.h"

// Add any new Loss* types to this list:
static const std::vector<std::shared_ptr<const RegisterLossBase>> RegisterLosses() {
   // IMPORTANT: the *LossParam types here must match the parameters types in your Loss* constructor
   return {
      RegisterLoss<LossMulticlassLogLoss>("log_loss"),
      RegisterLoss<LossRegressionPseudoHuber>("pseudo_huber", FloatLossParam("delta", 1))
   };
}

// TODO: include ranking
//
// We use the following terminology:
// Target      : the thing we're trying to predict.  For classification this is the label.  For regression this 
//               is what we're predicting directly.  Target and Output seem to be used interchangeably in other 
//               packages.  We choose Target here.
// Score       : the values we use to generate predictions.  For classification these are logits.  For regression these
//               are the predictions themselves.  For multiclass there are N scores per target when there are N classes.
//               For multiclass you could eliminate one score to get N-1 scores, but we don't use that in this package.
// Prediction  : the prediction of the model.  We output scores in our model and generate predictions from them.
//               For multiclass the scores are the logits, and the predictions would be the outputs of softmax.
//               We have N scores per target for an N class multiclass problem.
// Binary      : binary classification.  Target is 0 or 1
// Multiclass  : multiclass classification.  Target is 0, 1, 2, ... 
// Regression  : regression
// Multioutput : a model that can predict multiple different things.  A single model could predict binary, 
//               multiclass, regression, etc. different targets.
// Multitask   : A slightly more restricted form of multioutput where training jointly optimizes the targets.
//               The different targets can still be of different types (binary, multiclass, regression, etc), but
//               importantly they share a single loss function.  In C++ we deal only with multitask since otherwise 
//               it would make more sense to train the targets separately.  In higher level languages the models can 
//               either be Multitask or Multioutput depending on how they were generated.
// Multilabel  : A more restricted version of multitask where the tasks are all binary classification.  For 
//               consistency with our other types of Multitask learning, we call this MultitaskBinary*.
// 
// The most general loss function that we could handle in C++ would be to take a custom loss function that jointly 
// optimizes a multitask problem that contains regression, binary, and multiclass tasks.  This would be: 
// "LossMultitaskCustom"


// !! ANYTHING BELOW THIS POINT ISN'T REQUIRED TO MAKE YOUR OWN CUSTOM LOSS FUNCTION !!

ErrorEbmType Loss::CreateLoss(
   const char * const sLoss, 
   const Config * const pConfig,
   const Loss ** const ppLoss
) noexcept {
   EBM_ASSERT(nullptr != sLoss);
   EBM_ASSERT(nullptr != pConfig);
   EBM_ASSERT(nullptr != ppLoss);

   LOG_0(TraceLevelInfo, "Entered Loss::CreateLoss");
   
   try {
      const std::vector<std::shared_ptr<const RegisterLossBase>> registeredLosses = RegisterLosses();
      for(const std::shared_ptr<const RegisterLossBase> & lossRegistration : registeredLosses) {
         if(nullptr == lossRegistration) {
            // hopefully nobody inserts a nullptr, but check anyways
            LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss construction exception");
            return Error_LossConstructionException;
         }
         try {
            std::unique_ptr<const Loss> pLoss = lossRegistration->AttemptCreateLoss(*pConfig, sLoss);
            if(nullptr != pLoss) {
               // found it!
               LOG_0(TraceLevelInfo, "Exited Loss::CreateLoss");
               // we're exiting the area where exceptions are regularily thrown 
               *ppLoss = pLoss.release();
               return Error_None;
            }
         } catch(const SkipLossException &) {
            // the specific Loss function is saying it isn't a match (based on parameters in the Config object probably)
         }
      }
   } catch(const EbmException & exception) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Exception");
      return exception.GetError();
   } catch(const std::bad_alloc&) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss Out of Memory");
      return Error_OutOfMemory;
   } catch(...) {
      LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss construction exception");
      return Error_LossConstructionException;
   }
   
   LOG_0(TraceLevelWarning, "WARNING Loss::CreateLoss loss unknown");
   return Error_LossUnknown;
}

FloatEbmType Loss::GetUpdateMultiple() const {
   return FloatEbmType { 1 };
}
