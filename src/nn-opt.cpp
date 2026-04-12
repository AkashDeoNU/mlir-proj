//
// NN is a compiler that takes a neural net and transforms it into LLVM IR.
// It relies on four transformations from a high level language.
// 1. Lower NN. Func + NN -> Func + Linalg + Tensor + Arith
// (NNLowerToLinalg.cpp).
// 2. Bufferization. Func + Linalg + Tensor + Arith -> Func + Linalg + MemRef +
// Arith (MLIR).
// 3. LinalgToLoops. Func + Linalg + MemRef + Arith -> Func + SCF + MemRef +
// Arith (MLIR).
// 4. Lower To LLVM. Func + SCF + MemRef + Arith -> LLVM IR (NNLowerToLLVM.cpp).
//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "NNDialect.h"
#include "NNOps.h"
#include "NNLowerToLinalg.h"
#include "NNLowerToLLVM.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: nn-opt <input.mlir>\n";
    return 1;
  }

  // 1. Create the context and register dialects
  mlir::MLIRContext context;

  // Register BufferizableOpInterface external models before loading the
  // dialects that need them, so one-shot bufferize can find the impls.
  {
    mlir::DialectRegistry registry;
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
  }

  context.loadDialect<nn::NNDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::bufferization::BufferizationDialect>();

  // Register the LLVMIR translation interface for the llvm + builtin
  // dialects, so translateModuleToLLVMIR can lower them to a real
  // llvm::Module after the pass pipeline finishes.
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  // 2. Parse the input file into a module
  mlir::OwningOpRef<mlir::ModuleOp> module =
	mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // 3. Create a pass manager and add passes
  mlir::PassManager pm(&context);

  // Fold relu(relu(x)) -> relu(x) before we lose the nn.relu shape.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // NN -> Linalg + Tensor + Arith
  pm.addPass(std::make_unique<nn::NNToLinalgLoweringPass>());

  // tensor -> memref everywhere, including func signatures.
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // linalg.generic / linalg.max / linalg.fill on memrefs -> scf.for nests.
  pm.addPass(mlir::createConvertLinalgToLoopsPass());

  // Everything left (func + scf + cf + arith + memref) -> llvm dialect
  // in one shot, sharing an LLVMTypeConverter.
  pm.addPass(std::make_unique<nn::NNLowerToLLVMPass>());


  // 4. Run the pass manager
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  // 5. Translate the llvm-dialect module into a real llvm::Module and
  //    print it as textual LLVM IR.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }
  llvmModule->print(llvm::outs(), /*AAW=*/nullptr);
  return 0;
}
