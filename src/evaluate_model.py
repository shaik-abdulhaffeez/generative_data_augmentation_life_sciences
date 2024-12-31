import torch

def evaluate_model(classifier, X_test_tensor, y_test_tensor):
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(X_test_tensor)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        print(f'Accuracy of the classifier on the test set: {accuracy.item() * 100:.2f}%')