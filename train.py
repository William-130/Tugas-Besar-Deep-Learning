"""
Simple training script untuk FaceNet Face Recognition
"""

from facenet import FaceNetModel

def main():
    # ==================== CONFIGURATION ====================
    # Edit konfigurasi di sini sesuai kebutuhan
    
    # Dataset paths
    DATA_DIR = "Train/Train"
    
    # Training hyperparameters
    NUM_EPOCHS = 30  # Lebih banyak epoch untuk melihat konvergensi
    BATCH_SIZE = 16   # MINIMAL 2 untuk BatchNorm! Recommended: 4-32
    LEARNING_RATE = 0.0001  # Learning rate lebih kecil untuk training stabil
    VALIDATION_SPLIT = 0.2
    
    # Augmentation
    ENABLE_AUGMENTATION = True
    NUM_AUGMENTATIONS = 40  # 40 augmentasi per gambar = ~41x data (1 original + 40 augmented)
    # Device
    DEVICE = 'cuda'
    
    # ======================================================
    
    print("=" * 70)
    print("🚀 FaceNet Face Recognition Training")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Augmentation: {'Enabled' if ENABLE_AUGMENTATION else 'Disabled'}")
    print(f"   Device: {DEVICE}")
    
    # Initialize model
    print("\n📦 Initializing FaceNet model...")
    model = FaceNetModel(device=DEVICE)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {DATA_DIR}")
    embeddings, labels = model.load_dataset(DATA_DIR, augment=ENABLE_AUGMENTATION, num_augmentations=NUM_AUGMENTATIONS)
    
    # Train
    print(f"\n🎓 Training classifier...")
    history = model.train_classifier(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_split=VALIDATION_SPLIT
    )
    
    # Save model
    print("\n💾 Saving model...")
    model_path = model.save_model('./models')
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    print(f"📊 Final Results:")
    print(f"   Best Val Accuracy: {model.best_val_acc:.2f}%")
    print(f"   Model saved to: {model_path}")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Unique persons: {len(set(labels))}")
    print("=" * 70)

if __name__ == "__main__":
    main()
