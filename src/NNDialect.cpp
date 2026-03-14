#include "NNDialect.h"
#include "NNOps.h"

#include "NNDialect.cpp.inc"

void nn::NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NNOps.cpp.inc"
      >();
}
