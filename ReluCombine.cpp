// Fold Relu(Relu(x)) -> Relu(x)
struct SimplifyTwoRelus : public mlir::OpRewritePattern<ReluOp> {
  SimplifyTwoRelus(mlir::MLIRContext *context)
	: OpRewritePattern<ReluOp>(context, /*benefit=*/1) {}

  llvm::LogicalResult
  matchAndRewrite(ReluOp op,
                  mlir::PatternRewriter &rewriter) const override {
	// Look at the input of the current relu
	mlir::Value reluInput = op.getInput();
	ReluOp reluInputOp = reluInput.getDefiningOp<ReluOp>();

	if (!reluInputOp) {
	  return failure();
	}

	rewriter.replaceOp(op, {reluInputOp.getResult()});
	return success();
  }
};

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyTwoRelus>(context);
}
