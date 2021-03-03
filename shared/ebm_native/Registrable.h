// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

#ifndef REGISTRABLE_H
#define REGISTRABLE_H

class Registrable {

protected:

   Registrable() = default;

public:
   virtual ~Registrable() = default;
};

#endif // REGISTRABLE_H