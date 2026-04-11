#include "NNLowerToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace nn {

void NNLowerToLLVMPass::runOnOperation() {
  // Target: only the `llvm` dialect (and the top-level ModuleOp) is legal.
  // Anything else surviving the conversion is a bug.
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // One type converter shared by every populate-call below, so that
  // memref -> struct, function signatures, block args, and branch operands
  // all agree on the LLVM-level ABI.
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());

  // scf.for / scf.if -> cond_br and block args. No type converter needed
  // here: scf and cf are isomorphic on operand types.
  populateSCFToControlFlowConversionPatterns(patterns);

  // The type-converting conversions. Order among these does not matter —
  // they all run in one driver.
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Full conversion: fail loudly if anything outside the llvm dialect is
  // left. At this point in the pipeline we want to know about it.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

} // namespace nn
