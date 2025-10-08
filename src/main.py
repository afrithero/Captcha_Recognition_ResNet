from trainer import Trainer
from model import ResNet
import model_config
import argparse
import mlflow

def main():
    parser = argparse.ArgumentParser(description="Captcha ResNet with MLflow")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "predict"],
                        help="Which pipeline to run")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for evaluate/predict)")
    parser.add_argument("--uri", type=str, default=None,
                        help="Uri to the registered model (for evaluate/predict)")
    parser.add_argument("--sample_submission", type=str, default=None,
                        help="Sample submission csv (for predict)")
    parser.add_argument("--submission", type=str, default=None,
                        help="Output submission csv (for predict)")
    args = parser.parse_args()

    trainer = Trainer(
        model=ResNet,
        device=model_config.DEVICE,
        batch_size=model_config.BATCH_SIZE,
        epochs=model_config.EPOCHS,
        save_dir="../log",
        lr=model_config.LR,
    )

    if args.mode == "train":
        trainer.train()

    elif args.mode == "evaluate":
        trainer.evaluate(checkpoint_path=args.checkpoint, model_uri=args.uri)

    elif args.mode == "predict":
        trainer.predict(
            sample_submission_path=args.sample_submission,
            submission_path=args.submission,
            checkpoint_path=args.checkpoint
        )

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    main()