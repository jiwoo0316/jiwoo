import torch
import torch.nn as nn
import torch.optim as optim


def train_research_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 25,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.0,
    use_cosine_lr: bool = False,
):
    """
    Train / Validation을 분리해서 수행하고,
    Validation Loss가 가장 낮은 시점의 모델을 best checkpoint로 저장/복원함.

    [개선 2] v4에서 추가된 옵션:
    - label_smoothing: CrossEntropyLoss에 label smoothing 적용 (기본 0.0)
    - use_cosine_lr: True이면 Cosine Annealing LR Scheduler 사용
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # [개선 2] Label Smoothing 적용
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # [개선 2] Cosine Annealing LR Scheduler
    scheduler = None
    if use_cosine_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n[학습 시작] Device: {device}, Epochs: {num_epochs}")
    if label_smoothing > 0:
        print(f"  Label Smoothing: {label_smoothing}")
    if use_cosine_lr:
        print(f"  LR Scheduler: CosineAnnealingLR (T_max={num_epochs})")

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100

        # Validation Loss가 더 낮으면 best checkpoint 갱신
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # [개선 2] Scheduler step
        if scheduler is not None:
            scheduler.step()

        # 진행 상황 출력 (현재 LR도 함께 표시)
        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

    # 가장 좋은 validation 성능을 보인 시점의 모델 복원
    print(f"-> 가장 낮은 Val Loss({best_val_loss:.4f}) 모델을 복원합니다.\n")
    model.load_state_dict(best_model_state)
    return model