// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template <typename TFloat>
struct CrossEntropyMulticlassMultitaskLoss : public MulticlassMultitaskLoss {

   // This is the most general format that I could envision we'd handle as a non-custom loss function.
   // It's not clear that we can really handle it nicely, but I'm leaving a placeholder here to think about it.  
   // An example of this might include a prediction problem having 2 targets, with the first target having 3 classes
   // and the second target having 4 classes.  In a higher level language this might be represented as two separate
   // models, or a single model that contains two targets.  We'd want to expose this externally as two tensors that
   // can be visualized and predicted separately.  The two targets could be separated entirely.  The only reason
   // to combine them here is to use a single loss function between them in case they have correlations between them.
   // allowing for more information to be extracted when calculating the gradients and hessians and therefore when
   // splitting and during updating.
   // Internally here though, we want to mush the scores together from the separate targets and classes.  We
   // need all targets and all class scores to calculate the gradients and hessians and gains and updates, so
   // co-locating them in memory is advantageous.  In C++ we can stack them as an array of 3 + 4 = 7 scores together
   // within each cell of the tensors.  Our public interface should probably separate these into separate tensors
   // when we transition our C layer interface boundary.  In the higher level interface, these would be accessed as:
   // score[index_target][index_term][dimension1, dimension2, dimension3, ... , index_class]
   // whereas in C++ we'd store them as:
   // score[index_term][dimension1, dimension2, dimension3, ... , index_target;index_class]
   //
   // To do this properly, we'd need to accept from the caller a count of targets, and then have an array with the
   // count of classes for each target.  We can mirror that information here by using the special template overrides 
   // and do our softmax per-target.  We can't use the compiler version of the count of scores though since our
   // arrays are jagged, so it'll be an oddball in that we'll have a score count of 7 if we have 2 targets with 3
   // and 4 classes, so our count of scores will be 7, but we'll want to pass through either 0 or 1 for the count
   // of scores that we pass to the templated TLoss functions since we don't want to use the templated hard-coded
   // compiler optimized count of outputs

   // There's an even more general case of multi-task learning with the targets being a mix of 
   // regression, binary classification, and multiclass, but that would obviously require a custom loss function.

   // In terms of being able to use the compiler to optimize the number of scores, MulticlassMultitaskLoss is different
   // than MulticlassLoss*, BinaryMultitaskLoss*, and RegressionMultitaskLoss* because those other task types
   // have a last dimension that is uniform, which can therefore use the templating system to get complier optimized 
   // counts of scores, unlike this more general case that needs to be special cased since the last
   // array is a jagged one with different inner array sizes.

};
