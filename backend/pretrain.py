import time
from tqdm import tqdm
from backend.main_analyze import train_model

def pretrain():
    print("🚀 Starting model pretraining...")
    start_time = time.time()
    
    best_acc = train_model(show_progress=True)
    
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.2%}")
    print(f"⏱️  Training duration: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    pretrain()