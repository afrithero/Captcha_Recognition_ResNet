from trainer import Trainer
from model.CNN import ResNet
import argparse
import mlflow
import yaml
import torch

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Captcha ResNet with MLflow")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "predict"],
                        help="Which pipeline to run")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="Path to yaml config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for evaluate/predict)")
    parser.add_argument("--uri", type=str, default=None,
                        help="Uri to the registered model (for evaluate/predict)")
    parser.add_argument("--sample_submission", type=str, default=None,
                        help="Sample submission csv (for predict)")
    parser.add_argument("--submission", type=str, default=None,
                        help="Output submission csv (for predict)")
    
    args = parser.parse_args()
    config = load_config(args.config)

    trainer = Trainer(
        model=ResNet,
        device=torch.device(config["device"] if torch.cuda.is_available() else "cpu"),
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        save_freq=config["save_freq"],
        save_dir="../log",
        lr=config["lr"],
    )

    if args.mode == "train":
        trainer.train()

    elif args.mode == "evaluate":
        trainer.evaluate(checkpoint_path=args.checkpoint, model_uri=args.uri)

    elif args.mode == "predict":
        trainer.predict(
            sample_submission_path=args.sample_submission,
            submission_path=args.submission,
            checkpoint_path=args.checkpoint,
            model_uri=args.uri
        )

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    main()