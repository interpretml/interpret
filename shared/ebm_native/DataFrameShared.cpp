// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#include "PrecompiledHeader.h"

#include <stdlib.h> // free
#include <stddef.h> // size_t, ptrdiff_t
#include <string.h> // memcpy

#include "ebm_native.h" // FloatEbmType
#include "EbmInternal.h"
#include "logging.h" // EBM_ASSERT & LOG
#include "FeatureAtomic.h"
#include "FeatureGroup.h"

// TODO : create our shared dataframe here!