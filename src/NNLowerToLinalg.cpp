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

// Lower `nn.matmul %lhs, %rhs` to:
//   %init   = tensor.empty(%M, %N)
//   %zeroed = linalg.fill ins(0.0) outs(%init)   // matmul accumulates
//   %res    = linalg.matmul ins(%lhs, %rhs) outs(%zeroed)
struct MatmulOpLowering : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto resultType = cast<RankedTensorType>(op.getType());

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "matmul expects 2-D operands");

    Type f64 = rewriter.getF64Type();

    // Result is [M, N]: M comes from lhs dim 0, N from rhs dim 1.
    SmallVector<Value> dynSizes;
    if (resultType.isDynamicDim(0))
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, lhs, 0));
    if (resultType.isDynamicDim(1))
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, rhs, 1));

    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  f64, dynSizes);

    // linalg.matmul reads its output operand as the accumulator, so it
    // must be zeroed before the contraction starts.
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64FloatAttr(0.0));
    Value zeroed = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{zeroScalar},
                                               ValueRange{init})
                       .getResult(0);

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, /*inputs=*/ValueRange{lhs, rhs},
        /*outputs=*/ValueRange{zeroed});
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

// Lower `nn.add %lhs, %rhs` to a linalg.add. Both operands must have
// the same rank.
struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    Type f64 = rewriter.getF64Type();
    int64_t resRank = resultType.getRank();

    // Dynamic dim list for the result, derived from lhs (which always
    // has the full output rank — rhs may be lower-rank).
    SmallVector<Value> resDynSizes;
    for (int64_t i = 0; i < resRank; ++i) {
      if (resultType.isDynamicDim(i))
        resDynSizes.push_back(rewriter.create<tensor::DimOp>(loc, lhs, i));
    }

    auto makeEmpty = [&]() -> Value {
      return rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), f64,
                                              resDynSizes);
    };

    if (rhsType.getRank() != lhsType.getRank()) {
	  return rewriter.notifyMatchFailure(op, "add expects operands of equal rank");
    }

    Value addInit = makeEmpty();
    auto added = rewriter.create<linalg::AddOp>(
        loc, ValueRange{lhs, rhs}, ValueRange{addInit});
    rewriter.replaceOp(op, added.getResults());
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
  patterns.add<ReluOpLowering, MatmulOpLowering, AddOpLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

} // namespace nn
