"""
Face Recognition menggunakan FaceNet (Inception-ResNet)
FaceNet menghasilkan embeddings 512-dimensional untuk face recognition
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
from tqdm import tqdm


class FaceNetModel:
    """
    FaceNet Model wrapper untuk face recognition
    Menggunakan InceptionResnetV1 pretrained pada VGGFace2
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        # Load MTCNN untuk face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            device=torch.device('cpu'),  # MTCNN lebih stabil di CPU
            keep_all=False,
            post_process=True,
            thresholds=[0.4, 0.5, 0.5],
            min_face_size=20
        )
        print(f"✅ MTCNN loaded on CPU")
        
        # Load FaceNet (InceptionResnetV1)
        self.facenet = InceptionResnetV1(
            pretrained='vggface2',  # Pretrained pada VGGFace2 dataset
            classify=False,  # Tidak untuk klasifikasi langsung
            num_classes=None
        ).eval().to(self.device)
        
        print(f"✅ FaceNet (InceptionResnetV1) loaded on {self.device}")
        print(f"   Embedding dimension: 512")
        
        # Storage untuk training data
        self.embeddings = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
    def crop_face(self, image_path, debug=False):
        """Crop wajah dari gambar menggunakan MTCNN"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize jika gambar terlalu besar
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple([int(x * ratio) for x in image.size])
                image = image.resize(new_size, Image.LANCZOS)
            
            face = self.mtcnn(image)
            
            if debug and face is not None:
                print(f"  ✅ Detected face: {face.shape}")
            
            return face
        except Exception as e:
            if debug:
                print(f"  ❌ Error: {e}")
            return None
    
    def extract_embedding(self, face_tensor):
        """Extract 512-dim embedding menggunakan FaceNet"""
        if face_tensor is None:
            return None
        
        try:
            # FaceNet expects input shape: (batch, 3, 160, 160)
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            face_tensor = face_tensor.to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"  ❌ Embedding extraction error: {e}")
            return None
    
    def load_dataset(self, data_dir, augment=False, num_augmentations=10):
        """
        Load dataset dari folder
        Args:
            data_dir: Path ke folder berisi subfolder per person
            augment: Jika True, lakukan data augmentation
            num_augmentations: Jumlah augmentasi per gambar (default: 10)
        """
        print(f"\n🚀 Loading dataset from: {data_dir}")
        print("=" * 70)
        
        embeddings = []
        labels = []
        
        success_count = 0
        fail_count = 0
        
        # Iterasi setiap person folder
        for person_name in os.listdir(data_dir):
            person_folder = os.path.join(data_dir, person_name)
            
            if not os.path.isdir(person_folder):
                continue
            
            print(f"\n📂 Processing: {person_name}")
            
            # Iterasi setiap gambar
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                # Crop face
                face_tensor = self.crop_face(img_path)
                
                if face_tensor is not None:
                    # Extract embedding
                    embedding = self.extract_embedding(face_tensor)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        labels.append(person_name)
                        success_count += 1
                        
                        # Data augmentation (jika diaktifkan)
                        if augment:
                            aug_faces = self.augment_face(face_tensor, num_augmentations=num_augmentations)
                            for aug_face in aug_faces:
                                aug_embedding = self.extract_embedding(aug_face)
                                if aug_embedding is not None:
                                    embeddings.append(aug_embedding)
                                    labels.append(person_name)
                else:
                    fail_count += 1
                    print(f"  ⚠️  No face: {img_file}")
        
        print("\n" + "=" * 70)
        print(f"✅ Dataset loaded!")
        print(f"   Success: {success_count} faces")
        print(f"   Failed: {fail_count} images")
        print(f"   Total embeddings: {len(embeddings)}")
        print(f"   Unique persons: {len(set(labels))}")
        if augment:
            print(f"   Augmentation: {num_augmentations}x per image")
            print(f"   Data multiplier: ~{len(embeddings) / success_count:.1f}x")
        
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)
        
        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        return self.embeddings, self.labels
    
    def augment_face(self, face_tensor, num_augmentations=10):
        """
        Data augmentation untuk face tensor dengan lebih banyak variasi
        Args:
            face_tensor: Face tensor [C, H, W]
            num_augmentations: Jumlah augmentasi yang akan dibuat
        """
        import torchvision.transforms.functional as TF
        import random
        
        augmented = []
        
        # Pastikan tensor dalam range [0, 1]
        face_tensor = torch.clamp(face_tensor, 0, 1)
        face_pil = TF.to_pil_image(face_tensor.squeeze(0) if face_tensor.dim() == 4 else face_tensor)
        
        for _ in range(num_augmentations):
            aug_img = face_pil.copy()
            
            # Random kombinasi augmentasi
            # 1. Rotation (-15 sampai +15 derajat)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                aug_img = TF.rotate(aug_img, angle, fill=0)
            
            # 2. Horizontal Flip
            if random.random() > 0.5:
                aug_img = TF.hflip(aug_img)
            
            # 3. Brightness adjustment (0.7 - 1.3x)
            if random.random() > 0.5:
                brightness = random.uniform(0.7, 1.3)
                aug_img = TF.adjust_brightness(aug_img, brightness)
            
            # 4. Contrast adjustment (0.8 - 1.2x)
            if random.random() > 0.5:
                contrast = random.uniform(0.8, 1.2)
                aug_img = TF.adjust_contrast(aug_img, contrast)
            
            # 5. Saturation adjustment (0.8 - 1.2x)
            if random.random() > 0.5:
                saturation = random.uniform(0.8, 1.2)
                aug_img = TF.adjust_saturation(aug_img, saturation)
            
            # 6. Gaussian Blur (kadang-kadang)
            if random.random() > 0.7:  # 30% chance
                from PIL import ImageFilter
                aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
            
            augmented.append(TF.to_tensor(aug_img))
        
        return augmented
    
    def train_classifier(self, num_epochs=20, batch_size=32, learning_rate=1e-3, validation_split=0.2):
        """
        Train classifier head di atas FaceNet embeddings
        """
        print(f"\n🎓 Training classifier...")
        print("=" * 70)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.embeddings, self.labels,
            test_size=validation_split,
            random_state=42,
            stratify=self.labels if len(set(self.labels)) > 1 else None
        )
        
        print(f"📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Classes: {len(self.label_to_idx)}")
        
        # Create datasets
        train_dataset = EmbeddingDataset(X_train, y_train, self.label_to_idx)
        val_dataset = EmbeddingDataset(X_val, y_val, self.label_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        # Create classifier
        num_classes = len(self.label_to_idx)
        classifier = EmbeddingClassifier(
            embedding_dim=512, 
            num_classes=num_classes,
            dropout_rate=0.5  # Tingkatkan dropout untuk regularization
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization (weight decay)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5
        )
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"\n{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Status':<15}")
        print("=" * 90)
        
        for epoch in range(num_epochs):
            # Training
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for embeddings_batch, labels_batch in train_loader:
                embeddings_batch = embeddings_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                outputs = classifier(embeddings_batch)
                loss = criterion(outputs, labels_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for embeddings_batch, labels_batch in val_loader:
                    embeddings_batch = embeddings_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs = classifier(embeddings_batch)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels_batch.size(0)
                    val_correct += (predicted == labels_batch).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduler
            scheduler.step(val_acc)
            
            status = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                status = "✅ BEST"
            
            # Early stopping jika overfitting (train acc >> val acc)
            if train_acc - val_acc > 20:  # Gap > 20% = overfitting parah
                status += " ⚠️ OVERFITTING!"
            
            print(f"{epoch+1:<8} | {train_loss:<12.4f} | {train_acc:<12.2f} | {val_loss:<12.4f} | {val_acc:<12.2f} | {status:<15}")
        
        print("=" * 90)
        print(f"\n✅ Training completed!")
        print(f"   Best Epoch: {best_epoch}")
        print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        
        self.classifier = classifier
        self.history = history
        self.best_val_acc = best_val_acc
        
        return history
    
    def predict(self, image_path, threshold=0.6):
        """
        Predict identity dari gambar
        Args:
            image_path: Path ke gambar
            threshold: Threshold similarity untuk classification
        Returns:
            (predicted_name, similarity_score)
        """
        # Crop face
        face_tensor = self.crop_face(image_path)
        
        if face_tensor is None:
            return "No Face Detected", 0.0
        
        # Extract embedding
        test_embedding = self.extract_embedding(face_tensor)
        
        if test_embedding is None:
            return "Embedding Failed", 0.0
        
        # Compare dengan training embeddings
        test_embedding_reshaped = test_embedding.reshape(1, -1)
        similarities = cosine_similarity(test_embedding_reshaped, self.embeddings)[0]
        
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= threshold:
            predicted_name = self.labels[max_idx]
        else:
            predicted_name = "Unknown"
        
        return predicted_name, max_similarity
    
    def save_model(self, save_dir='./models'):
        """Save model dan embeddings"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"facenet_model_{timestamp}.pkl")
        
        # Save classifier jika ada
        classifier_path = None
        if hasattr(self, 'classifier'):
            classifier_path = os.path.join(save_dir, f"facenet_classifier_{timestamp}.pth")
            torch.save(self.classifier.state_dict(), classifier_path)
        
        # Save data
        model_data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'timestamp': timestamp,
            'embedding_dim': 512,
            'model_type': 'FaceNet-InceptionResnetV1'
        }
        
        if hasattr(self, 'history'):
            model_data['history'] = self.history
            model_data['best_val_acc'] = self.best_val_acc
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Model saved!")
        print(f"   Embeddings: {model_path}")
        if classifier_path:
            print(f"   Classifier: {classifier_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """Load model dari file"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.embeddings = model_data['embeddings']
        self.labels = model_data['labels']
        self.label_to_idx = model_data['label_to_idx']
        self.idx_to_label = model_data['idx_to_label']
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"   Total embeddings: {len(self.embeddings)}")
        print(f"   Unique persons: {len(self.label_to_idx)}")
        
        return model_data


class EmbeddingDataset(Dataset):
    """Dataset untuk embeddings"""
    def __init__(self, embeddings, labels, label_to_idx):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor([label_to_idx[label] for label in labels])
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EmbeddingClassifier(nn.Module):
    """Classifier head untuk embeddings dengan regularization lebih kuat"""
    def __init__(self, embedding_dim=512, num_classes=None, dropout_rate=0.5):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization untuk stabilitas
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize FaceNet model
    model = FaceNetModel(device='cuda')
    
    # Load dataset
    data_dir = "../../train/train"
    embeddings, labels = model.load_dataset(data_dir, augment=True, num_augmentations=15)
    
    # Train classifier
    history = model.train_classifier(
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        validation_split=0.2
    )
    
    # Save model
    model_path = model.save_model('./models')
    
    # Test prediction
    test_image = "../test/1.jpg"
    predicted_name, similarity = model.predict(test_image, threshold=0.6)
    print(f"\n🎯 Prediction: {predicted_name}")
    print(f"📊 Similarity: {similarity:.4f}")
