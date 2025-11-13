import argparse
from nlp.spacy_model import trainSpacyTextcat, evaluateSpacy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train CSV")
    ap.add_argument("--dev", default=None, help="Optional dev CSV; if not given, will split from train")
    ap.add_argument("--textCol", default="text")
    ap.add_argument("--labelCol", default="label")
    ap.add_argument("--nEpochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batchSize", type=int, default=64)
    ap.add_argument("--outputDir", default="nlp/spacy_model/artifacts")
    ap.add_argument("--test", default=None, help="Optional test CSV for evaluation after training")
    args = ap.parse_args()

    best_path = trainSpacyTextcat(
        trainCsvPath=args.train,
        textCol=args.textCol,
        labelCol=args.labelCol,
        nEpochs=args.nEpochs,
        lr=args.lr,
        dropout=args.dropout,
        batchSize=args.batchSize,
        outputDir=args.outputDir,
        devCsvPath=args.dev
    )

    if args.test:
        metrics = evaluateSpacy(
            testCsvPath=args.test,
            textCol=args.textCol,
            labelCol=args.labelCol,
            modelPath=best_path
        )
        print("=== Test Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
