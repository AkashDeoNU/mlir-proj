#include "NNLowerToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace nn {

namespace {

// Lower `nn.relu %x` to:
//   %zi   = tensor.empty(...)                        // init for the fill
//   %z    = linalg.fill ins(%c0 : f64) outs(%zi)     // zero tensor
//   %ri   = tensor.empty(...)                        // init for the max
//   %res  = linalg.max ins(%x, %z) outs(%ri)
struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto tensorType = cast<RankedTensorType>(input.getType());
    Type f64 = rewriter.getF64Type();
    int64_t rank = tensorType.getRank();

    // Every tensor.empty we build needs to match the runtime shape of
    // %input, so capture its dynamic dims once.
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (tensorType.isDynamicDim(i))
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }

    auto makeEmpty = [&]() -> Value {
      return rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(), f64,
                                              dynSizes);
    };

    // Zero tensor: empty + fill with scalar 0.0.
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64FloatAttr(0.0));
    Value zeroInit = makeEmpty();
    Value zeros = rewriter
                      .create<linalg::FillOp>(loc, ValueRange{zeroScalar},
                                              ValueRange{zeroInit})
                      .getResult(0);

    // Elementwise max(input, zeros) into a fresh init.
    Value maxInit = makeEmpty();
    auto maxed = rewriter.create<linalg::MaxOp>(
        loc, /*inputs=*/ValueRange{input, zeros},
        /*outputs=*/ValueRange{maxInit});

    rewriter.replaceOp(op, maxed.getResults());
    return success();
  }
};

} // namespace

void NNToLinalgLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, linalg::LinalgDialect,
                         arith::ArithDialect, tensor::TensorDialect>();
  target.addIllegalDialect<NNDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ReluOpLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

} // namespace nn
